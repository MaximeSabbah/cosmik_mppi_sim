{
  description = "ROS integration for Franka research robots";

  inputs = {
    nix-ros-overlay.url = "github:lopsided98/nix-ros-overlay/master";
    libfranka-common = {
      # tingelst fork because
      # https://github.com/frankaemika/libfranka-common/issues/1
      url = "github:tingelst/libfranka-common";
      flake = false;
    };
  };

  outputs =
    { libfranka-common, nix-ros-overlay, self, ... }:
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
          default = self.packages.${system}.libfranka;
          libfranka = pkgs.rosPackages.humble.libfranka.overrideAttrs {
            src = pkgs.lib.fileset.toSource {
              root = ./.;
              fileset = pkgs.lib.fileset.unions [
                ./cmake
                ./doc
                ./examples
                ./include
                ./scripts
                ./src
                ./test
                ./CHANGELOG.md
                ./CMakeLists.txt
                ./LICENSE
                ./NOTICE
                ./README.md
              ];
            };
            # CMake has a `add_subdirectory(common)` which is a git submodule
            preConfigure = ''
              ln -s ${libfranka-common} common
            '';
          };
        };
      }
    );
}
