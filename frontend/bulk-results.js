(function (root) {
    "use strict";

    function escapeHtml(value) {
        return String(value).replace(/[&<>"']/g, character => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            "\"": "&quot;",
            "'": "&#39;"
        })[character]);
    }

    function renderBulkRows(results) {
        return results.map(item => {
            const detections = item.detections || [];
            if (!detections.length) {
                return `<tr class="low-row"><td>${escapeHtml(item.filename)}</td>` +
                    `<td>${item.count || 0}</td>` +
                    `<td colspan="3" class="warning">No Detection</td></tr>`;
            }

            const boxes = detections.map(d => {
                const b = d.bbox;
                return `<li>(${b.x1.toFixed(1)}, ${b.y1.toFixed(1)}, ` +
                    `${b.x2.toFixed(1)}, ${b.y2.toFixed(1)})</li>`;
            }).join("");
            const confidences = detections.map(d =>
                `<li>${(d.confidence * 100).toFixed(1)}%</li>`
            ).join("");
            const statuses = detections.map(d => {
                const status = d.review_required ? "Review Needed" : "Accepted";
                const className = d.review_required ? "warning" : "success";
                return `<li class="${className}">${status}</li>`;
            }).join("");

            return `<tr class="${detections.some(d => d.review_required) ? "low-row" : ""}">` +
                `<td>${escapeHtml(item.filename)}</td><td>${item.count}</td>` +
                `<td><ol>${boxes}</ol></td><td><ol>${confidences}</ol></td>` +
                `<td><ol>${statuses}</ol></td></tr>`;
        }).join("");
    }

    function csvEscape(value) {
        return `"${String(value ?? "").replaceAll('"', '""')}"`;
    }

    function buildBulkCsv(results) {
        const rows = [[
            "Filename", "Detection Count", "Detection #", "x1", "y1", "x2", "y2",
            "Confidence", "Status"
        ]];

        results.forEach(item => {
            const detections = item.detections || [];
            if (!detections.length) {
                rows.push([
                    item.filename, item.count || 0, "", "", "", "", "", "",
                    "No Detection"
                ]);
                return;
            }
            detections.forEach((d, index) => rows.push([
                item.filename,
                item.count,
                index + 1,
                d.bbox.x1,
                d.bbox.y1,
                d.bbox.x2,
                d.bbox.y2,
                `${(d.confidence * 100).toFixed(2)}%`,
                d.review_required ? "Review Needed" : "Accepted"
            ]));
        });

        return rows.map(row => row.map(csvEscape).join(",")).join("\r\n") + "\r\n";
    }

    const api = {escapeHtml, renderBulkRows, buildBulkCsv};
    if (typeof module !== "undefined" && module.exports) module.exports = api;
    else root.BulkResults = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
