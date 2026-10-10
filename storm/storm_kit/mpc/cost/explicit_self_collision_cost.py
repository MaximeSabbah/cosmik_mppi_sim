"""Conservative FER sphere-cover self collision with explicit SRDF exclusions."""
from pathlib import Path
import json,re,xml.etree.ElementTree as ET
import numpy as np,torch,yaml
from ...util_file import get_assets_path,get_mpc_configs_path,join_path

class ExplicitSelfCollisionCost(torch.nn.Module):
    def __init__(self, weight, robot_params, tensor_args, distance_threshold=.01,
                 srdf_path=None, explicit_geometry=True, **unused):
        super().__init__()
        self.weight=float(weight);self.margin=float(distance_threshold)
        if self.margin<=0:raise ValueError('Explicit self margin must be positive')
        cfg=robot_params['robot_collision_params'];path=Path(join_path(get_assets_path(),cfg['collision_spheres']))
        if not path.is_file():
            # Native sphere YAML paths historically resolve from configs/mpc.
            path = Path(join_path(get_mpc_configs_path(), cfg['collision_spheres']))
        spheres=yaml.safe_load(path.read_text())['collision_spheres'];coverage=json.loads((path.parent/'coverage.json').read_text())['coverage']
        if not srdf_path:raise ValueError('Explicit self geometry requires an explicit SRDF path')
        srdf_path = Path(join_path(get_assets_path(), str(srdf_path)))
        disabled={frozenset((x.attrib['link1'],x.attrib['link2'])) for x in ET.parse(srdf_path).findall('disable_collisions')}
        links=cfg['link_objs'];centers=[];radii=[];parents=[];sources=[]
        for parent,link in enumerate(links):
            names=[]
            for item in coverage:
                if item['link']==link:
                    match=re.match(r'fer_(?:link\d+|hand|leftfinger|rightfinger)',item['source'])
                    if not match:raise ValueError('Unrecognized FER coverage source '+item['source'])
                    names.extend([match.group()]*item['spheres'])
            if len(names)!=len(spheres[link]):raise ValueError('Sphere coverage provenance mismatch for '+link)
            for sphere,source in zip(spheres[link],names):
                centers.append(sphere['center']);radii.append(sphere['radius']);parents.append(parent);sources.append(source)
        pairs=[(i,j) for i in range(len(sources)) for j in range(i+1,len(sources)) if sources[i]!=sources[j] and frozenset((sources[i],sources[j])) not in disabled]
        if not pairs:raise ValueError('No explicit self collision pairs')
        for name,value in [('centers',centers),('radii',radii)]:self.register_buffer(name,torch.tensor(value,**tensor_args))
        for name,value in [('parents',parents),('first',[i for i,j in pairs]),('second',[j for i,j in pairs])]:self.register_buffer(name,torch.tensor(value,device=tensor_args['device'],dtype=torch.long))
        self.pair_count=len(pairs);self._kernel=None

    def __getstate__(self):
        state=super().__getstate__();state['_kernel']=None;return state

    def _distance(self, pos, rot):
        world=(rot[...,self.parents,:,:] @ self.centers.unsqueeze(-1)).squeeze(-1)+pos[...,self.parents,:]
        delta=world[...,self.first,:]-world[...,self.second,:]
        distances=torch.linalg.vector_norm(delta,dim=-1)-self.radii[self.first]-self.radii[self.second]
        return distances.amin(dim=-1)

    def signed_clearance(self, link_pos_seq,link_rot_seq):
        if self._kernel is None:self._kernel=torch.compile(self._distance,fullgraph=True,dynamic=False)
        with torch.autocast(device_type=link_pos_seq.device.type,enabled=False):
            return self._kernel(link_pos_seq.to(self.centers.dtype),link_rot_seq.to(self.centers.dtype))

    def forward(self,q,link_pos_seq=None,link_rot_seq=None,return_violation=False):
        if link_pos_seq is None or link_rot_seq is None:raise ValueError('Explicit self geometry needs link poses')
        clearance=self.signed_clearance(link_pos_seq,link_rot_seq)
        violation=torch.relu((self.margin-clearance)/self.margin)
        cost=self.weight*violation.square()
        return (cost,violation) if return_violation else cost
