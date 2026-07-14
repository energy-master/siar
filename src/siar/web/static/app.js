// Vixen Intelligence c.2026
//
// The dashboard. Vanilla ES modules, no framework, no build step.
//
// The one idea worth knowing: the spectrogram PNG is exactly `frames` x `n_bins` pixels —
// one image pixel per grid cell — and the SVG overlay shares that coordinate system via its
// viewBox. So a detection is drawn from its GRID coordinates (frame_lo, bin_lo, ...) with no
// conversion, no scaling maths, and no drift when the image is resized by CSS. The browser
// never needs to know what an FFT is.

const state = {
  runs: [],
  files: [],
  detections: [],
  run: null,
  file: null,
  minScore: 0
}

const $ = (id) => document.getElementById(id)

const fetchJSON = async (url) => {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`${response.status} ${url}`)
  return response.json()
}

const fmtTime = (s) => `${s.toFixed(2)} s`
const fmtHz = (hz) => (hz >= 1000 ? `${(hz / 1000).toFixed(1)} kHz` : `${Math.round(hz)} Hz`)
const fmtScore = (v) => (v >= 1000 ? v.toExponential(1) : v.toFixed(1))

// --- rendering ---------------------------------------------------------------

const renderRuns = () => {
  const list = $('runs')
  list.innerHTML = ''

  if (!state.runs.length) {
    $('runs-hint').textContent = 'no runs yet'
    return
  }
  $('runs-hint').hidden = true

  state.runs.forEach((run) => {
    const li = document.createElement('li')
    li.className = 'run' + (state.run?.run_uid === run.run_uid ? ' active' : '')
    li.tabIndex = 0
    li.setAttribute('role', 'button')
    li.setAttribute('aria-label', `Run ${run.name}, ${run.n_detections} detections`)
    li.innerHTML = `
      <span class="run-name">${run.name}</span>
      <span class="run-meta">${run.n_files} files · ${run.n_detections} detections</span>
      <span class="run-uid">${run.run_uid}</span>`

    const handleSelect = () => selectRun(run.run_uid)
    li.addEventListener('click', handleSelect)
    li.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        handleSelect()
      }
    })
    list.appendChild(li)
  })
}

const renderSummary = () => {
  const run = state.run
  const box = $('run-summary')
  if (!run) {
    box.hidden = true
    return
  }
  const hours = state.files.reduce((a, f) => a + f.duration_s, 0) / 3600
  const perHour = hours > 0 ? (run.n_detections / hours).toFixed(1) : '0.0'

  box.hidden = false
  box.innerHTML = `
    <div class="summary-grid">
      <div><dt>model</dt><dd>${run.model_name} <span class="dim">(${run.detector})</span></dd></div>
      <div><dt>corpus</dt><dd class="mono">${run.input_path}</dd></div>
      <div><dt>threshold</dt><dd>z = ${run.threshold.toFixed(2)}</dd></div>
      <div><dt>detections</dt><dd>${run.n_detections} <span class="dim">(${perHour}/hour)</span></dd></div>
    </div>
    <a class="download" href="/api/runs/${run.run_uid}/export.json"
       aria-label="Download this run as JSON">Download JSON</a>`
}

const renderFiles = () => {
  const list = $('files')
  list.innerHTML = ''

  if (!state.files.length) {
    $('files-hint').textContent = 'no recordings'
    $('files-hint').hidden = false
    return
  }
  $('files-hint').hidden = true

  // Already sorted most-anomalous-first by the API — that is the order to triage in.
  state.files.forEach((file) => {
    const li = document.createElement('li')
    li.className = 'file' + (state.file?.file_id === file.file_id ? ' active' : '')
    li.tabIndex = 0
    li.setAttribute('role', 'button')
    li.setAttribute('aria-label', `${file.name}, ${file.n_detections} detections`)

    const score = file.max_score === null ? '—' : fmtScore(file.max_score)
    li.innerHTML = `
      <span class="file-name">${file.name}</span>
      <span class="badge${file.n_detections ? ' hot' : ''}">${file.n_detections}</span>
      <span class="file-meta">top ${score}</span>`

    const handleSelect = () => selectFile(file.file_id)
    li.addEventListener('click', handleSelect)
    li.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        handleSelect()
      }
    })
    list.appendChild(li)
  })
}

const visibleDetections = () =>
  state.detections.filter((d) => d.score >= state.minScore)

const renderOverlay = () => {
  const file = state.file
  const svg = $('overlay')
  svg.innerHTML = ''
  if (!file) return

  // The whole trick: the viewBox IS the grid. x = frames, y = bins.
  svg.setAttribute('viewBox', `0 0 ${file.frames} ${file.n_bins}`)
  if (!$('show-boxes').checked) return

  const ns = 'http://www.w3.org/2000/svg'
  visibleDetections().forEach((d, i) => {
    const rect = document.createElementNS(ns, 'rect')
    rect.setAttribute('x', d.frame_lo)
    // SVG y grows downward and the image is flipped so high frequency is at the top, so a
    // detection at bins [lo, hi] sits at y = n_bins - 1 - hi.
    rect.setAttribute('y', file.n_bins - 1 - d.bin_hi)
    rect.setAttribute('width', d.frame_hi - d.frame_lo + 1)
    rect.setAttribute('height', d.bin_hi - d.bin_lo + 1)
    rect.setAttribute('class', 'box')
    rect.setAttribute('data-index', i)

    const title = document.createElementNS(ns, 'title')
    title.textContent =
      `#${i + 1}  ${fmtTime(d.t_start)}–${fmtTime(d.t_end)}  ` +
      `${fmtHz(d.f_low)}–${fmtHz(d.f_high)}  score ${fmtScore(d.score)}`
    rect.appendChild(title)

    rect.addEventListener('mouseenter', () => highlightRow(i, true))
    rect.addEventListener('mouseleave', () => highlightRow(i, false))
    svg.appendChild(rect)
  })
}

const highlightRow = (index, on) => {
  const row = $('detections').querySelector(`tbody tr[data-index="${index}"]`)
  if (row) row.classList.toggle('lit', on)
}

const renderDetections = () => {
  const body = $('detections').querySelector('tbody')
  body.innerHTML = ''

  const shown = visibleDetections()
  $('det-hint').textContent = shown.length
    ? `${shown.length} of ${state.detections.length} detections shown`
    : 'no detections above this score'

  shown.forEach((d, i) => {
    const tr = document.createElement('tr')
    tr.dataset.index = i
    tr.innerHTML = `
      <td class="dim">${i + 1}</td>
      <td>${fmtTime(d.t_start)}</td>
      <td>${fmtTime(d.t_end)}</td>
      <td>${fmtHz(d.f_low)}</td>
      <td>${fmtHz(d.f_high)}</td>
      <td class="score">${fmtScore(d.score)}</td>
      <td class="dim">${d.fill.toFixed(2)}</td>`

    tr.addEventListener('mouseenter', () => pulseBox(i, true))
    tr.addEventListener('mouseleave', () => pulseBox(i, false))
    body.appendChild(tr)
  })
}

const pulseBox = (index, on) => {
  const rect = $('overlay').querySelector(`rect[data-index="${index}"]`)
  if (rect) rect.classList.toggle('lit', on)
}

const renderViewer = () => {
  const file = state.file
  if (!file) {
    $('viewer').hidden = true
    return
  }
  $('viewer').hidden = false
  $('empty').hidden = true

  $('file-title').textContent = file.name
  $('spec-img').src = `/api/files/${file.file_id}/spectrogram.png`
  $('spec-img').alt = `Spectrogram of ${file.name}`
  $('ax-t0').textContent = '0 s'
  $('ax-t1').textContent = `${file.t_max.toFixed(1)} s`
  $('ax-f').textContent = `${fmtHz(file.f_min)} – ${fmtHz(file.f_max)} ↑`

  const top = state.detections.length ? state.detections[0].score : 100
  const slider = $('score-filter')
  slider.max = Math.ceil(top)
  slider.value = 0
  state.minScore = 0
  $('score-value').textContent = '0'

  renderOverlay()
  renderDetections()
}

// --- actions -----------------------------------------------------------------

const selectRun = async (uid) => {
  state.run = await fetchJSON(`/api/runs/${uid}`)
  state.files = await fetchJSON(`/api/runs/${uid}/files`)
  state.file = null
  state.detections = []

  renderRuns()
  renderSummary()
  renderFiles()
  $('viewer').hidden = true
  $('empty').hidden = false

  // Jump straight to the most anomalous recording — it is what the user came for.
  const first = state.files.find((f) => f.n_detections > 0) || state.files[0]
  if (first) selectFile(first.file_id)
}

const selectFile = async (fileId) => {
  state.file = state.files.find((f) => f.file_id === fileId) || null
  state.detections = await fetchJSON(`/api/files/${fileId}/detections`)
  renderFiles()
  renderViewer()
}

const handleScoreFilter = (event) => {
  state.minScore = Number(event.target.value)
  $('score-value').textContent = fmtScore(state.minScore)
  renderOverlay()
  renderDetections()
}

const handleToggleBoxes = () => renderOverlay()

// --- boot --------------------------------------------------------------------

const boot = async () => {
  $('score-filter').addEventListener('input', handleScoreFilter)
  $('show-boxes').addEventListener('change', handleToggleBoxes)

  try {
    const health = await fetchJSON('/api/health')
    $('health').textContent = `v${health.version}`
  } catch {
    $('health').textContent = 'offline'
  }

  state.runs = await fetchJSON('/api/runs')
  renderRuns()
  if (state.runs.length) selectRun(state.runs[0].run_uid)
}

boot()
