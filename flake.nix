{
  description = "Manim development environment (uv workflow)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            uv
            python312
            ruff
            pyright
          ];

          env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (
            with pkgs;
            [
              stdenv.cc.cc.lib
              zlib
              libGL
              libxcb
              xorg.libX11
              glib
            ]
          );
        };
      }
    );
}
