{
  description = "ROS integration for Franka research robots";

  inputs.nix-ros-overlay.url = "github:lopsided98/nix-ros-overlay/master";

  outputs =
    { nix-ros-overlay, self, ... }:
    nix-ros-overlay.inputs.flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nix-ros-overlay.inputs.nixpkgs {
          inherit system;
          overlays = [ nix-ros-overlay.overlays.default ];
        };
      in
      {
        packages = {
          default = self.packages.${system}.franka-description;
          franka-description = pkgs.rosPackages.humble.franka-description.overrideAttrs {
            src = pkgs.lib.fileset.toSource {
              root = ./.;
              fileset = pkgs.lib.fileset.unions [
                ./CMakeLists.txt
                ./end_effectors
                ./env-hooks
                ./launch
                ./meshes
                ./package.xml
                ./robots
                ./rviz
                ./scripts
                ./test
                ./worlds
              ];
            };
          };
        };
      }
    );
}
