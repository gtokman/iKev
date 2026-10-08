import Foundation
import Testing

@testable import Kev

/// Downloads the published checkpoint from the Hub and makes one decision. Opt-in: set `KEV_HUB` to the repo id
/// (e.g. `gtokman/iKev`); it needs network and ~0.8 GB of cache, so regular test runs skip it.
@Suite(.serialized) struct KevHubTests {
    static var hubID: String? { ProcessInfo.processInfo.environment["KEV_HUB"] }

    @Test func downloadsAndDecides() async throws {
        guard let hubID = Self.hubID else { return }
        let started = Date()
        let model = try await KevModel.load(hubID: hubID)
        print("hub load \(hubID): \(String(format: "%.1f", Date().timeIntervalSince(started))) s")
        let router = KevRouter(model: model, threshold: 0)
        let decision = try #require(try await router.route(latest: "set a timer for 10 minutes"))
        print("route: \(decision.route) confidence \(String(format: "%.2f", decision.confidence))")
        #expect(decision.route == .deviceAction)
    }
}
