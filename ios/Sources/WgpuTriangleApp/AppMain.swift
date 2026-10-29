import UIKit

// Declare the external C ABI symbol exported by our Rust static library
@_silgen_name("run_app")
func run_app()

@main
struct AppMain {
    static func main() {
        // Hands control over to winit's EventLoop, which initializes
        // UIApplicationMain and handles the iOS application lifecycle.
        run_app()
    }
}
