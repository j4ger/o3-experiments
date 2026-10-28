{
  description = "Android reverse-engineering shell for the Xiaomi / XRING NPU investigation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              # Device / packaging
              android-tools
              apktool
              jadx
              jdk17
              scrcpy

              # Native library inspection / PoC build
              binutils
              patchelf
              clang
              cmake
              ninja
              pkg-config

              # General tooling
              python3
              uv
              jq
              ripgrep
              file
              unzip
              zip
            ];

            shellHook = ''
              # Keep adb keys/config inside the project instead of ~/.android.
              export ANDROID_USER_HOME="$PWD/.android"
              echo "Android RE shell"
              echo "  adb:    $(command -v adb)"
              echo "  apktool: $(command -v apktool)"
              echo "  jadx:   $(command -v jadx)"
            '';
          };
        });
    };
}
