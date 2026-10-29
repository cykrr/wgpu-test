// swift-tools-version: 6.0
import PackageDescription
import Foundation

let repoRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent() // removes Package.swift
    .deletingLastPathComponent() // removes ios/
    .path

let rustLibDirDebug = "\(repoRoot)/target/aarch64-apple-ios/debug"
let rustLibDirRelease = "\(repoRoot)/target/aarch64-apple-ios/release"

let package = Package(
    name: "WgpuTriangleApp",
    platforms: [
        .iOS(.v17),
    ],
    products: [
        .library(
            name: "WgpuTriangleApp",
            targets: ["WgpuTriangleApp"]
        ),
    ],
    targets: [
        .target(
            name: "WgpuTriangleApp",
            linkerSettings: [
                .unsafeFlags([
                    "-L\(rustLibDirRelease)",
                    "-L\(rustLibDirDebug)",
                    "-lwgpu_triangle",
                ]),
                .linkedFramework("Metal"),
                .linkedFramework("QuartzCore"),
                .linkedFramework("UIKit"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("Foundation"),
            ]
        ),
    ]
)
