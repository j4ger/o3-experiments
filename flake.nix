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
          pkgs = import nixpkgs {
            inherit system;
            config.android_sdk.accept_license = true;
            config.allowUnfree = true;
          };
          ndk = pkgs.androidenv.androidPkgs.ndk-bundle;
          ndkPackages = pkgs.lib.optionals (system == "x86_64-linux") [ ndk ];
        in
        {
          default = pkgs.mkShell {
            packages = (with pkgs; [
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
            ]) ++ ndkPackages ++ (with pkgs; [
              # General tooling
              python3
              uv
              jq
              ripgrep
              file
              unzip
              zip
            ]);

            shellHook = ''
              # Keep adb keys/config inside the project instead of ~/.android.
              export ANDROID_USER_HOME="$PWD/.android"
              ${pkgs.lib.optionalString (system == "x86_64-linux") ''
                export NDK="${ndk}/libexec/android-sdk/ndk-bundle"
              ''}
              echo "Android RE shell"
              echo "  adb:    $(command -v adb)"
              echo "  apktool: $(command -v apktool)"
              echo "  jadx:   $(command -v jadx)"
              echo "  ndk:    $NDK"
            '';
          };
        });

      packages = forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.android_sdk.accept_license = true;
            config.allowUnfree = true;
          };
          ndk = pkgs.androidenv.androidPkgs.ndk-bundle;
          probe = pkgs.stdenv.mkDerivation {
            pname = "probe-npu-matmul";
            version = "0.1.0";
            src = ./.;

            nativeBuildInputs = [ ndk ];
            dontStrip = true;
            dontPatchELF = true;
            dontFixup = true;

            buildPhase = ''
              runHook preBuild
              ndk=${ndk}/libexec/android-sdk/ndk-bundle
              cc="$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang"
              cxx="$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++"
              mkdir -p build
              "$cc" -O2 -std=c11 -Wall -Wextra -I tools/npu \
                -c tools/npu/o3npu.c -o build/o3npu.o
              "$cxx" -O2 -std=c++17 -Wall -Wextra -I tools/npu \
                -c tools/npu/xnnl_shim.cpp -o build/xnnl_shim.o
              "$cxx" -O2 -std=c++17 -Wall -Wextra -I tools/npu \
                tools/probe/probe_npu_matmul.cpp \
                build/o3npu.o build/xnnl_shim.o -ldl \
                -o build/probe_npu_matmul
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              install -Dm755 build/probe_npu_matmul $out/bin/probe_npu_matmul
              runHook postInstall
            '';

            meta = with pkgs.lib; {
              description = "Arm64 Android probe for the XRING NPU matmul / GemmSplitMN path";
              platforms = [ "x86_64-linux" ];
            };
          };
        in
        if system == "x86_64-linux" then {
          probe-npu-matmul = probe;
        } else {});
    };
}
