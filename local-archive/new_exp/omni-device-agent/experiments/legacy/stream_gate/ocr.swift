import Foundation
import Vision
import AppKit

// One JSON line per image. Reference-only OCR; never a gating input.
for path in CommandLine.arguments.dropFirst() {
    do {
        guard let image = NSImage(contentsOfFile: path),
              let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw NSError(domain: "image", code: 1)
        }
        let req = VNRecognizeTextRequest()
        req.recognitionLevel = .accurate
        req.usesLanguageCorrection = false
        req.recognitionLanguages = ["en-US"]
        let start = ProcessInfo.processInfo.systemUptime
        try VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])
        let rows: [[String: Any]] = (req.results ?? []).compactMap { item in
            guard let c = item.topCandidates(1).first else { return nil }
            return ["text": c.string, "confidence": c.confidence,
                    "bbox": [item.boundingBox.minX, item.boundingBox.minY,
                             item.boundingBox.width, item.boundingBox.height]]
        }
        let result: [String: Any] = ["path": path, "rows": rows,
            "seconds": ProcessInfo.processInfo.systemUptime - start]
        print(String(data: try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]), encoding: .utf8)!)
    } catch {
        fputs("OCR failed: \(path): \(error)\n", stderr)
        exit(1)
    }
}
