/**
 * The behaviour of `siar-app serve`'s page.
 *
 * One rule shapes all of it: fetch what is on screen and nothing else. A survey folder can hold ten
 * thousand recordings, so the index arrives a page at a time, a lane's thumbnail is fetched when
 * the row scrolls into view, and a recording's boxes and picture are fetched only when it is
 * opened. Nothing is pulled through the tunnel speculatively, and the audio is not pulled at all
 * until somebody presses play.
 *
 * The token is read once from `?t=` and sent two ways: as a query parameter on <img> and <audio>
 * sources, which cannot carry a header, and as `X-Siar-Token` on every fetch, so the header path
 * the IDent Dynamics client will use is exercised by the page that ships with the daemon.
 *
 * VIXEN INTELLIGENCE, c. 2026
 */

const TOKEN = new URLSearchParams(location.search).get('t') || ''
const PAGE = 200

/** Colours per shape, so a box in the picture and its chip in the legend agree. */
const SHAPE_COLOURS = {
  sweep: '#7AD151', tonal: '#22A884', click: '#FDE725', click_train: '#F0616D',
  patch: '#35B779', blob: '#31688E', broadband: '#D29922',
}
const shapeColour = (shape) => SHAPE_COLOURS[shape] || '#A6B8BE'

const $ = (id) => document.getElementById(id)

/** A URL on this daemon, with the token in the query — for <img>, <audio> and links. */
const url = (route, params = {}) => {
  const u = new URL(route.replace(/^\//, ''), document.baseURI)
  for (const [key, value] of Object.entries(params)) {
    if (value !== null && value !== undefined && value !== '') u.searchParams.set(key, value)
  }
  if (TOKEN) u.searchParams.set('t', TOKEN)
  return u.toString()
}

/** A JSON fetch, tokenised by header rather than by query. */
const api = async (route, params = {}) => {
  const u = new URL(route.replace(/^\//, ''), document.baseURI)
  for (const [key, value] of Object.entries(params)) {
    if (value !== null && value !== undefined && value !== '') u.searchParams.set(key, value)
  }
  const response = await fetch(u, { headers: { 'X-Siar-Token': TOKEN }, credentials: 'omit' })
  if (!response.ok) {
    let code = String(response.status)
    try { code = (await response.json()).error || code } catch { /* not JSON; keep the status */ }
    throw new Error(code)
  }
  return await response.json()
}

// -- formatting, matching what the CLI prints ----------------------------------------------

const duration = (sec) => {
  const s = Number(sec) || 0
  if (s >= 3600) return `${(s / 3600).toFixed(2)} h`
  if (s >= 60) return `${(s / 60).toFixed(1)} min`
  return `${s.toFixed(1)} s`
}
const cost = (sec) => (Number(sec) < 1 ? `${(Number(sec) * 1000).toFixed(0)} ms` : duration(sec))
const clock = (sec) => {
  const whole = Math.max(0, Math.floor(Number(sec) || 0))
  const h = Math.floor(whole / 3600), m = Math.floor((whole % 3600) / 60), s = whole % 60
  const mm = String(m).padStart(2, '0'), ss = String(s).padStart(2, '0')
  return h ? `${h}:${mm}:${ss}` : `${m}:${ss}`
}
const factor = (f) => (Number(f) > 0 ? (f >= 10 ? `${f.toFixed(1)}x` : `${f.toFixed(2)}x`) : '—')
const count = (n) => (Number(n) || 0).toLocaleString()
const bytes = (n) => {
  let size = Number(n) || 0
  for (const unit of ['B', 'KiB', 'MiB', 'GiB']) {
    if (size < 1024 || unit === 'GiB') return `${unit === 'B' ? size : size.toFixed(1)} ${unit}`
    size /= 1024
  }
  return `${size.toFixed(1)} GiB`
}
const hz = (v) => (v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(Math.round(v)))
const ago = (stamp) => {
  const then = Date.parse(stamp || '')
  if (Number.isNaN(then)) return ''
  return `as of ${duration(Math.max(0, (Date.now() - then) / 1000))} ago`
}

// -- state ---------------------------------------------------------------------------------

const state = {
  meta: null,
  rows: [],
  total: 0,
  offset: 0,
  picked: null,
  boxes: [],
  header: null,
  hidden: new Set(),
  order: 'asc',
}

// -- the masthead --------------------------------------------------------------------------

const paintMeta = (meta) => {
  state.meta = meta
  $('folder').textContent = meta.folder || 'folder'
  $('readonly').hidden = false

  if (meta.state === 'no-manifest') {
    $('run').textContent = 'no run manifest yet — this folder has not flushed one'
    $('progress').hidden = true
    return
  }

  const algorithm = meta.algorithm || {}
  const totals = meta.totals || {}
  const stft = meta.stft || {}
  const parts = [
    `<b>${algorithm.slug || 'scan'}</b>${algorithm.version ? ` v${algorithm.version}` : ''}`,
    `${count(totals.files)} recordings`,
    duration(totals.audio_sec),
    `<b>${count(totals.structures)}</b> structures`,
    `fft ${stft.fft} / hop ${stft.hop}`,
    `${meta.workers || 1} worker${meta.workers === 1 ? '' : 's'}`,
  ]
  if (totals.files !== meta.index_covers) {
    // The manifest describes a run, not a census of the folder: say so rather than let somebody
    // wonder where the rest of their survey went.
    parts.push(`<span class="dim">index covers ${count(meta.index_covers)}</span>`)
  }
  $('run').innerHTML = parts.join('<span class="sep">·</span>')

  const progress = meta.progress || {}
  const running = meta.state === 'running'
  $('progress').hidden = !running
  if (running) {
    const fraction = Math.max(0, Math.min(1, Number(progress.fraction) || 0))
    $('progress-fill').style.width = `${(fraction * 100).toFixed(1)}%`
    $('progress-done').textContent =
      `${count(progress.files_done)} / ${count(progress.files_total)} files`
    $('progress-left').textContent =
      progress.eta_sec === null || progress.eta_sec === undefined
        ? 'estimating' : `${clock(progress.eta_sec)} left`
    // The manifest is rewritten on a duty cycle, so this page can outrun the writer. Saying how
    // old the numbers are is better than implying they are live.
    $('progress-age').textContent = ago(progress.updated_at)
  }

  fillOptions($('status'), Object.keys(totals.by_status || {}), 'any outcome')
  fillOptions($('shape'), Object.keys(totals.shapes || {}), 'any shape')
}

const fillOptions = (select, values, blank) => {
  if (select.dataset.filled === values.join(',')) return
  select.dataset.filled = values.join(',')
  const chosen = select.value
  select.innerHTML = `<option value="">${blank}</option>` +
    values.map((v) => `<option value="${v}">${v.replace(/_/g, ' ')}</option>`).join('')
  select.value = chosen
}

// -- the lane strip ------------------------------------------------------------------------

/** Thumbnails are fetched when their row appears, and never for the ten thousand that have not. */
const thumbs = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue
    const img = entry.target
    thumbs.unobserve(img)
    img.src = url('/api/thumbnail', { path: img.dataset.path })
    img.addEventListener('error', () => {
      // No thumbnail — too short to render, or --no-thumbnails. One fallback, then give up.
      if (img.dataset.fellBack) { img.classList.add('blank'); img.removeAttribute('src'); return }
      img.dataset.fellBack = '1'
      img.src = url('/api/preview', { path: img.dataset.path, w: 200, h: 64 })
    }, { once: true })
  }
}, { rootMargin: '200px' })

const laneNode = (row) => {
  const lane = document.createElement('div')
  lane.className = 'lane'
  lane.setAttribute('role', 'listitem')
  lane.tabIndex = 0
  lane.dataset.path = row.path

  // The lane is right-to-left so a long path loses its start rather than its filename. That
  // reorders the path's own punctuation — `134250533.191114211719.wav` comes out `wav.134250533…`
  // — unless the text is isolated from the container's direction, which is what <bdi> is for.
  const path = document.createElement('div')
  path.className = 'lane-path'
  const name = document.createElement('bdi')
  name.textContent = row.path
  path.append(name)
  path.title = row.path

  const badge = document.createElement('div')
  badge.className = 'lane-count'
  badge.textContent = row.status === 'scanned' ? count(row.structures) : ''

  const thumb = document.createElement('img')
  thumb.className = 'lane-thumb'
  thumb.alt = ''
  thumb.loading = 'lazy'
  thumb.dataset.path = row.path
  if (row.thumbnail || row.status === 'scanned') thumbs.observe(thumb)
  else thumb.classList.add('blank')

  const foot = document.createElement('div')
  foot.className = 'lane-foot'
  const bits = [`<span class="status-${row.status}">${row.status.replace(/_/g, ' ')}</span>`,
    duration(row.duration_sec)]
  if (row.status === 'scanned') bits.push(factor(row.realtime_factor))
  for (const [shape, n] of Object.entries(row.shapes || {})) {
    bits.push(`<span class="shape" style="color:${shapeColour(shape)}">${shape} ${count(n)}</span>`)
  }
  if (row.error) bits.push(`<span class="status-error">${row.error.slice(0, 60)}</span>`)
  foot.innerHTML = bits.join(' ')

  lane.append(path, badge, thumb, foot)
  lane.addEventListener('click', () => pick(row))
  lane.addEventListener('keydown', (e) => { if (e.key === 'Enter') pick(row) })
  return lane
}

const loadIndex = async (append = false) => {
  const params = {
    offset: append ? state.offset : 0,
    limit: PAGE,
    q: $('q').value.trim(),
    status: $('status').value,
    shape: $('shape').value,
    sort: $('sort').value,
    order: state.order,
  }
  let page
  try {
    page = await api('/api/index', params)
  } catch (e) {
    $('counted').textContent = e.message === 'no_manifest'
      ? 'no run manifest yet' : `could not read the index (${e.message})`
    return
  }
  if (!append) { $('lanes').innerHTML = ''; state.rows = [] }
  state.total = page.total
  state.offset = page.offset + page.files.length
  state.rows.push(...page.files)

  const fragment = document.createDocumentFragment()
  for (const row of page.files) fragment.append(laneNode(row))
  $('lanes').append(fragment)

  $('counted').textContent = `${count(state.rows.length)} of ${count(page.total)}`
  $('more').hidden = state.offset >= page.total
  markPicked()
}

const markPicked = () => {
  for (const lane of $('lanes').children) {
    lane.setAttribute('aria-current', String(lane.dataset.path === state.picked?.path))
  }
}

// -- one recording -------------------------------------------------------------------------

const pick = async (row) => {
  state.picked = row
  state.boxes = []
  state.hidden = new Set()
  markPicked()
  $('empty').hidden = true
  $('picked').hidden = false
  $('picked-path').textContent = row.path
  $('tip').hidden = true

  const meta = [
    `<span class="status-${row.status}">${row.status.replace(/_/g, ' ')}</span>`,
    duration(row.duration_sec),
    `${count(row.structures)} structures`,
    `scanned in ${cost(row.elapsed_sec)}`,
    factor(row.realtime_factor),
    bytes(row.audio_bytes),
  ]
  $('picked-meta').innerHTML = meta.join('<span class="sep">·</span>')

  paintAudio(row)
  paintPreview(row)
  await paintBoxes(row)
}

const paintPreview = (row) => {
  const img = $('preview')
  const width = Math.min(4000, Math.max(400, Math.round(
    ($('plot').clientWidth || 900) * (window.devicePixelRatio || 1))))
  $('plot-note').textContent = 'drawing…'
  img.onload = () => {
    $('plot-note').textContent = ''
    state.header = {
      width: Number(img.naturalWidth),
      duration: Number(row.duration_sec),
      nyquist: state.header?.nyquist || 0,
    }
    sizeCanvas()
    drawBoxes()
    paintAxes(row)
  }
  img.onerror = () => {
    $('plot-note').textContent = row.status === 'too_short'
      ? 'too short to picture' : 'no preview for this recording'
    img.removeAttribute('src')
  }
  img.src = url('/api/preview', { path: row.path, w: width, h: 256 })
}

const paintAxes = (row) => {
  const nyquist = state.nyquist || 0
  const rows = []
  for (let i = 4; i >= 0; i -= 1) {
    rows.push(`<span>${nyquist ? `${hz((nyquist * i) / 4)}Hz` : ''}</span>`)
  }
  $('axis-y').innerHTML = rows.join('')
  const seconds = Number(row.duration_sec) || 0
  $('axis-x').innerHTML = [0, 0.25, 0.5, 0.75, 1]
    .map((f) => `<span>${duration(seconds * f)}</span>`).join('')
}

const paintBoxes = async (row) => {
  if (row.status !== 'scanned' || !row.structures) { drawBoxes(); paintLegend(); return }
  try {
    const document_ = await api('/api/structures', { path: row.path })
    if (state.picked?.path !== row.path) return  // the reader moved on while this was in flight
    state.boxes = Array.isArray(document_.structures) ? document_.structures : []
    state.nyquist = Number((document_.stft || {}).sample_rate || 0) / 2
    paintAxes(row)
    drawBoxes()
    paintLegend()
  } catch (e) {
    $('plot-note').textContent = `could not read the boxes (${e.message})`
  }
}

const sizeCanvas = () => {
  const canvas = $('boxes')
  const rect = $('plot').getBoundingClientRect()
  const ratio = window.devicePixelRatio || 1
  canvas.width = Math.max(1, Math.round(rect.width * ratio))
  canvas.height = Math.max(1, Math.round(rect.height * ratio))
}

const drawBoxes = () => {
  const canvas = $('boxes')
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  const seconds = Number(state.picked?.duration_sec) || 0
  const nyquist = state.nyquist || 0
  if (!seconds || !nyquist) return

  ctx.lineWidth = Math.max(1, (window.devicePixelRatio || 1))
  for (const box of state.boxes) {
    if (state.hidden.has(box.shape)) continue
    const x = (Number(box.tmin) / seconds) * canvas.width
    const w = Math.max(2, ((Number(box.tmax) - Number(box.tmin)) / seconds) * canvas.width)
    // The picture has the lowest frequency at the bottom, so a box's top edge is fmax.
    const y = (1 - Number(box.fmax) / nyquist) * canvas.height
    const h = Math.max(2, ((Number(box.fmax) - Number(box.fmin)) / nyquist) * canvas.height)
    ctx.strokeStyle = shapeColour(box.shape)
    ctx.strokeRect(x, y, w, h)
  }
}

const paintLegend = () => {
  const counts = {}
  for (const box of state.boxes) counts[box.shape] = (counts[box.shape] || 0) + 1
  const shapes = Object.entries(counts).sort((a, b) => b[1] - a[1])
  $('legend').innerHTML = shapes.map(([shape, n]) => `
    <button type="button" data-shape="${shape}" aria-pressed="true">
      <span class="swatch" style="background:${shapeColour(shape)}"></span>${shape} ${count(n)}
    </button>`).join('')
  for (const button of $('legend').children) {
    button.addEventListener('click', () => {
      const shape = button.dataset.shape
      const on = button.getAttribute('aria-pressed') === 'true'
      button.setAttribute('aria-pressed', String(!on))
      if (on) state.hidden.add(shape); else state.hidden.delete(shape)
      drawBoxes()
    })
  }
}

/** Hovering a box says what it is, which is the whole reason to draw them over the picture. */
$('boxes').addEventListener('mousemove', (event) => {
  const canvas = $('boxes')
  const rect = canvas.getBoundingClientRect()
  const seconds = Number(state.picked?.duration_sec) || 0
  const nyquist = state.nyquist || 0
  if (!seconds || !nyquist) return
  const t = ((event.clientX - rect.left) / rect.width) * seconds
  const f = (1 - (event.clientY - rect.top) / rect.height) * nyquist
  const hit = state.boxes.find((b) => !state.hidden.has(b.shape)
    && t >= b.tmin && t <= b.tmax && f >= b.fmin && f <= b.fmax)
  const tip = $('tip')
  if (!hit) { tip.hidden = true; return }
  tip.hidden = false
  tip.innerHTML = [
    `<b style="color:${shapeColour(hit.shape)}">${hit.shape}</b>`,
    `${Number(hit.tmin).toFixed(3)}–${Number(hit.tmax).toFixed(3)} s`,
    `${hz(hit.fmin)}–${hz(hit.fmax)} Hz`,
    `peak ${hz(hit.peakHz)} Hz`,
    `${count(hit.cells)} cells`,
    `confidence ${Number(hit.confidence).toFixed(2)}`,
  ].join('<span class="sep">·</span>')
})

const paintAudio = (row) => {
  const box = $('audio')
  if (!state.meta?.capabilities?.audio) {
    box.innerHTML = '<span class="dim">the audio route is disabled on this daemon</span>'
    return
  }
  const link = `<a href="${url('/api/audio', { path: row.path })}" download>download
    ${bytes(row.audio_bytes)}</a>`
  // preload="none" is load-bearing: nothing crosses the tunnel until somebody presses play. Above
  // a few tens of megabytes not even the element is offered, because Safari has ignored that
  // attribute before and a 900 MB autoload would saturate the link.
  const playable = row.audio_bytes > 0 && row.audio_bytes <= 64 * 1024 * 1024
  box.innerHTML = playable
    ? `${link}<audio controls preload="none" src="${url('/api/audio', { path: row.path })}"></audio>`
    : `${link} <span class="dim">— too large to stream from here; open the folder in IDent
       Dynamics for that</span>`
}

// -- the panels ----------------------------------------------------------------------------

const paintPerformance = async () => {
  let doc
  try {
    doc = await api('/api/performance')
  } catch (e) {
    $('performance').innerHTML = `<span class="dim">no performance report (${e.message})</span>`
    return
  }
  const totals = doc.totals || {}
  const machine = doc.machine || {}
  const phases = totals.phases || {}
  const measured = Object.values(phases).reduce((sum, v) => sum + (Number(v) || 0), 0)
  const workers = Number(machine.workers) || 1
  const parallel = workers > 1
  const total = parallel ? measured : Math.max(Number(totals.wall_sec) || 0, measured)

  const stages = Object.entries(phases).sort((a, b) => b[1] - a[1])
  if (!parallel && (totals.wall_sec || 0) > measured) {
    stages.push(['overhead', (Number(totals.wall_sec) || 0) - measured])
  }

  const rows = [
    ['audio scanned', duration(totals.audio_sec)],
    ['wall time', duration(totals.wall_sec)],
    ...(parallel ? [[`worker time (${workers} workers)`, duration(measured)]] : []),
  ].map(([label, value]) => `<tr><th>${label}</th><td class="num">${value}</td><td></td></tr>`)

  const stageRows = stages.map(([name, seconds]) => {
    const pct = total > 0 ? (100 * seconds) / total : 0
    return `<tr class="stage"><th>${name}</th><td class="num">${cost(seconds)}</td>
      <td><span class="share" style="width:${pct.toFixed(1)}%"></span></td>
      <td class="num">${pct.toFixed(0)}%</td></tr>`
  })

  const tail = [
    ['realtime', factor(totals.realtime_factor)],
    ['per recording', `${(Number(totals.sec_per_file) || 0).toFixed(2)} s`],
    ['machine', `${machine.platform || '—'} · ${machine.cpus || '?'} cores`],
  ].map(([label, value]) => `<tr><th>${label}</th><td class="num">${value}</td><td></td></tr>`)

  $('performance').innerHTML = `<table>${rows.join('')}${stageRows.join('')}${tail.join('')}</table>
    <p class="dim">${parallel
      ? 'Stage times are worker seconds — several workers spent them at once, so they add up to '
        + 'more than the wall clock. Shares are of the work.'
      : 'Shares are of wall time.'}</p>`
}

const paintRuns = async () => {
  let doc
  try {
    doc = await api('/api/runs')
  } catch (e) {
    $('runs').innerHTML = `<span class="dim">could not list runs (${e.message})</span>`
    return
  }
  const rows = (doc.runs || []).map((run) => {
    const where = run.current
      ? '<span class="here">serving this one</span>'
      : (run.exists
        ? `<span class="dim">siar-app serve ${run.out}</span>`
        : '<span class="dim">no longer on disk</span>')
    return `<tr><th>${run.name}</th><td class="num">${count(run.files)} files</td>
      <td class="num">${count(run.structures)}</td><td>${run.algorithm}</td><td>${where}</td></tr>`
  })
  $('runs').innerHTML = rows.length
    ? `<div class="runs"><table>${rows.join('')}</table></div>
       <p class="dim">One daemon serves one folder — restart it with another path to look at that
       one.</p>`
    : '<span class="dim">no runs recorded on this machine</span>'
}

// -- wiring --------------------------------------------------------------------------------

let debounce = 0
const refresh = () => {
  clearTimeout(debounce)
  debounce = setTimeout(() => loadIndex(false), 150)
}
$('q').addEventListener('input', refresh)
for (const id of ['status', 'shape', 'sort']) $(id).addEventListener('change', () => loadIndex(false))
$('order').addEventListener('click', () => {
  state.order = state.order === 'asc' ? 'desc' : 'asc'
  $('order').textContent = state.order === 'asc' ? '↓' : '↑'
  loadIndex(false)
})
$('more').addEventListener('click', () => loadIndex(true))
window.addEventListener('resize', () => { sizeCanvas(); drawBoxes() })
$('performance-panel').addEventListener('toggle', function once() {
  if (this.open) { this.removeEventListener('toggle', once); paintPerformance() }
})
$('runs-panel').addEventListener('toggle', function once() {
  if (this.open) { this.removeEventListener('toggle', once); paintRuns() }
})

const poll = async () => {
  try {
    const meta = await api('/api/meta')
    const wasRunning = state.meta?.state === 'running'
    const before = state.meta?.totals?.files
    paintMeta(meta)
    // A run in progress grows its own index, so reload the strip when the file count moves.
    if (wasRunning && meta.totals?.files !== before) loadIndex(false)
    if (meta.state === 'running') setTimeout(poll, 3000)
  } catch (e) {
    $('run').textContent = `lost the daemon (${e.message})`
  }
}

const start = async () => {
  if (!TOKEN) { $('nokey').hidden = false; return }
  await poll()
  await loadIndex(false)
}

start()
