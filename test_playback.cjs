// Exercise the browser-side state transitions without requiring a browser.
const assert = require('node:assert/strict');
global.window = {dash_clientside: {callback_context: {triggered: []}}};
require('./assets/playback.js');
const runs = [{history: Array(101).fill({})}];
function update(triggers, current, paused, stride=1) {
    window.dash_clientside.callback_context.triggered = triggers.map(id => ({prop_id: id+'.value'}));
    return window.dash_clientside.optimizationLab.playback(runs, 1, 1, 1, 1, current, paused, stride);
}
assert.deepEqual(update(['play'], 100, true), [100, 0, false, '❚❚ Pause']);
assert.deepEqual(update(['clock'], 0, false), [100, 1, false, '❚❚ Pause']);
assert.deepEqual(update(['clock', 'play'], 17, false), [100, 17, true, '▶ Play']);
assert.deepEqual(update(['clock'], 17, true), [100, 17, true, '▶ Play']);
assert.deepEqual(update(['play'], 17, true), [100, 17, false, '❚❚ Pause']);
assert.deepEqual(update(['step', 'clock'], 17, false), [100, 18, true, '▶ Play']);
assert.deepEqual(update(['final'], 17, false), [100, 100, true, '▶ Play']);
assert.deepEqual(update(['clock'], 99, false, 5), [100, 100, true, '▶ Play']);
assert.deepEqual(update(['runs'], 17, false), [100, 100, true, '▶ Play']);
console.log('Playback: start, pause, simultaneous tick, resume, step and completion verified.');
