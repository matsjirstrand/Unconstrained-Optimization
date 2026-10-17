/* Keep transport state in the browser: interval requests must not overwrite Pause. */
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    optimizationLab: {
        playback: function (runs, play, step, final, tick, current, paused, stride) {
            const maximum = Math.max(0, ...(runs || []).map(r => r.history.length - 1));
            current = Math.max(0, Math.min(maximum, current || 0));
            const changed = new Set((window.dash_clientside.callback_context.triggered || [])
                .map(t => t.prop_id.split('.')[0]));
            // Explicit user controls take priority over a simultaneous timer tick.
            if (changed.has('runs') || changed.has('final')) {
                current = maximum;
                paused = true;
            } else if (changed.has('play')) {
                if (paused && maximum > 0) {
                    if (current >= maximum) current = 0;
                    paused = false;
                } else {
                    paused = true;
                }
            } else if (changed.has('step')) {
                current = Math.min(maximum, current + 1);
                paused = true;
            } else if (changed.has('clock') && !paused) {
                const increment = Math.max(1, Math.min(1000, Math.floor(Number(stride) || 1)));
                current = Math.min(maximum, current + increment);
                paused = current >= maximum;
            }
            return [maximum, current, !!paused, paused ? '▶ Play' : '❚❚ Pause'];
        }
    }
});
