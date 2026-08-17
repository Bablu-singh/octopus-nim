/* Static-demo shim for GitHub Pages.
 *
 * Pages serves files, not Python, so there is no /api to call. Rather than fork the UI
 * into a second, drifting copy, this intercepts fetch and replays a dispatch that really
 * happened — captured from the live NVIDIA NIM API, timings and all. The page above is
 * byte-for-byte the app's own octopus.html, so what you see here is what it does locally.
 *
 * No API key is involved: the recording is finished text, and nothing calls out.
 */
(function () {
  const data = fetch("demo.json").then(r => r.json());
  const realFetch = window.fetch.bind(window);
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  // Replay speed. The real run took 165s; 3x keeps it watchable without misrepresenting
  // the ordering, overlap, or which agent finished when.
  window.DEMO_SPEED = 3;

  window.fetch = async function (url, opts) {
    const u = String(url);

    if (u.includes("/api/catalog")) {
      const d = await data;
      return new Response(JSON.stringify(d.catalog),
        {status: 200, headers: {"Content-Type": "application/json"}});
    }

    if (u.includes("/api/dispatch")) {
      const d = await data;
      const enc = new TextEncoder();
      const stream = new ReadableStream({
        async start(c) {
          let last = 0;
          for (const [ms, ev] of d.events) {
            const gap = (ms - last) / window.DEMO_SPEED;
            last = ms;
            if (gap > 0) await sleep(gap);
            c.enqueue(enc.encode("data: " + JSON.stringify(ev) + "\n\n"));
          }
          c.close();
        }
      });
      return new Response(stream, {status: 200, headers: {"Content-Type": "text/event-stream"}});
    }

    return realFetch(url, opts);
  };

  // Pin the task box to the recorded task: any other text would still replay this run.
  data.then(d => {
    const box = document.getElementById("task");
    if (!box) return;
    box.value = d.task;
    box.readOnly = true;
    box.title = "This demo replays one recorded dispatch, so the task is fixed.";
    const hint = document.getElementById("hint");
    if (hint) hint.textContent = "recorded run · " + Math.round(d.duration_ms / 1000) +
      "s of real time, replayed at " + window.DEMO_SPEED + "x";
  });
})();
