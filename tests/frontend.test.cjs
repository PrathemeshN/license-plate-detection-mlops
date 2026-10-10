const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const {buildBulkCsv, renderBulkRows} = require("../frontend/bulk-results.js");

const detection = (confidence, x1, review_required = false) => ({
    confidence,
    review_required,
    bbox: {x1, y1: 2, x2: 30, y2: 40}
});

test("bulk table renders every detection coordinate, confidence, and status", () => {
    const html = renderBulkRows([{
        filename: "plate.png",
        count: 2,
        detections: [detection(0.9, 1), detection(0.25, 5, true)]
    }]);
    assert.match(html, /\(1\.0, 2\.0, 30\.0, 40\.0\)/);
    assert.match(html, /\(5\.0, 2\.0, 30\.0, 40\.0\)/);
    assert.match(html, /90\.0%/);
    assert.match(html, /Review Needed/);
});

test("no detections remain explicit in table and CSV", () => {
    assert.match(renderBulkRows([{filename: "empty.png", count: 0, detections: []}]), /No Detection/);
    assert.match(buildBulkCsv([{filename: "empty.png", count: 0, detections: []}]), /No Detection/);
});

test("CSV writes one row per detection and retains review-needed status", () => {
    const csv = buildBulkCsv([{
        filename: "two.png",
        count: 2,
        detections: [detection(0.9, 1), detection(0.2, 5, true)]
    }]);
    assert.equal(csv.trimEnd().split("\r\n").length, 3);
    assert.match(csv, /90\.00%/);
    assert.match(csv, /Review Needed/);
    assert.match(csv, /"two.png","2","2","5","2","30","40","20\.00%","Review Needed"/);
});

test("CSV quotes filename commas, quotes, and newlines", () => {
    const csv = buildBulkCsv([{
        filename: 'plate, "front"\n2.png',
        count: 0,
        detections: []
    }]);
    assert.match(csv, /"plate, ""front""\n2\.png"/);
});

test("HTML escapes untrusted filenames", () => {
    const html = renderBulkRows([{filename: "<img src=x onerror=alert(1)>", count: 0, detections: []}]);
    assert.doesNotMatch(html, /<img/);
    assert.match(html, /&lt;img/);
});

test("served frontend inline JavaScript parses and loads the tested helpers", () => {
    const html = fs.readFileSync("frontend/index.html", "utf8");
    const inlineScript = html.match(/<script>([\s\S]*?)<\/script>/);
    assert.ok(inlineScript);
    assert.doesNotThrow(() => new vm.Script(inlineScript[1]));
    assert.match(html, /<script src="\/static\/bulk-results\.js"><\/script>/);
});
