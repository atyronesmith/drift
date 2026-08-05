/**
 * Canvas renderer — draws pipelines, gates, packets, and labels.
 * Supports multiple pipelines on separate lanes.
 * Hoverable elements register hit zones for DOM tooltip popovers.
 */

import { CHECKPOINT_TYPES } from './models.js'

const C = {
  bg:         '#0f172a',
  lane:       '#1e293b',
  laneStroke: '#334155',
  glow:       '#38bdf8',
  core:       '#ffffff',
  prod:       '#3b82f6',
  text:       '#e2e8f0',
  dim:        '#94a3b8',
  bright:     '#f8fafc',
  sep:        '#475569',
}

let canvas, ctx, canvasH = 500
let tooltip, hitZones = []

export function initRenderer(el, height) {
  canvas = el
  ctx = canvas.getContext('2d')
  if (height) canvasH = height

  // Create tooltip element
  const wrap = canvas.parentElement
  wrap.style.position = 'relative'
  tooltip = document.createElement('div')
  tooltip.className = 'canvas-tooltip'
  tooltip.style.cssText = 'display:none;position:absolute;pointer-events:none;z-index:10;'
  wrap.appendChild(tooltip)

  // Mouse tracking for tooltips
  canvas.addEventListener('mousemove', onMouseMove)
  canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none' })

  resize()
  window.addEventListener('resize', resize)
}

function onMouseMove(e) {
  const rect = canvas.getBoundingClientRect()
  const mx = e.clientX - rect.left
  const my = e.clientY - rect.top

  for (const zone of hitZones) {
    if (mx >= zone.x && mx <= zone.x + zone.w && my >= zone.y && my <= zone.y + zone.h) {
      tooltip.innerHTML = zone.html
      // Position tooltip above the zone, centered
      let tx = zone.x + zone.w / 2
      let ty = zone.y - 8
      tooltip.style.display = 'block'
      tooltip.style.left = tx + 'px'
      tooltip.style.top = ty + 'px'
      tooltip.style.transform = 'translate(-50%, -100%)'
      canvas.style.cursor = 'default'
      return
    }
  }
  tooltip.style.display = 'none'
  canvas.style.cursor = 'default'
}

export function resize() {
  const r = canvas.parentElement.getBoundingClientRect()
  const w = r.width
  const h = canvasH
  const d = devicePixelRatio || 1
  canvas.width = w * d
  canvas.height = h * d
  canvas.style.height = h + 'px'
  ctx.setTransform(d, 0, 0, d, 0, 0)
  return { w, h }
}

export function getSize() {
  const d = devicePixelRatio || 1
  return { w: canvas.width / d, h: canvas.height / d }
}

export function clear() {
  const { w, h } = getSize()
  ctx.fillStyle = C.bg
  ctx.fillRect(0, 0, w, h)
  hitZones = []
}

function addHitZone(x, y, w, h, html) {
  hitZones.push({ x, y, w, h, html })
}

function fmtTokens(n) {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return `${n}`
}

export function drawLane(pipeline) {
  const { w, h } = getSize()
  const y = pipeline.laneY * h
  ctx.strokeStyle = C.laneStroke
  ctx.lineWidth = 2
  ctx.setLineDash([6, 4])
  ctx.beginPath()
  ctx.moveTo(w * 0.04, y)
  ctx.lineTo(w * 0.98, y)
  ctx.stroke()
  ctx.setLineDash([])
}

/** Draw a label on the left side of a lane (no subtext — description is in popover). */
export function drawLaneLabel(pipeline, label, color) {
  const { w, h } = getSize()
  const y = pipeline.laneY * h
  ctx.fillStyle = color || C.dim
  ctx.font = 'bold 16px "DM Sans", system-ui, sans-serif'
  ctx.textAlign = 'left'
  ctx.fillText(label, w * 0.01, y - 50)

  // Measure text width for hit zone
  const tw = ctx.measureText(label).width
  const lx = w * 0.01
  const ly = y - 64

  addHitZone(lx, ly, tw, 20, `<strong>${pipeline.model.name}</strong><br>${pipeline.model.description}<br><span class="tt-dim">${pipeline.model.source}</span>`)
}

export function drawStepLabels(pipeline) {
  // no-op for single-step pipelines
}

export function drawGates(pipeline, activeGateIdx) {
  for (let i = 0; i < pipeline.gates.length; i++) {
    const g = pipeline.gates[i]
    const active = i === activeGateIdx
    const past = i < activeGateIdx

    if (g.isLLM || g.isAgent) {
      drawLLMBlock(g, active, past)
      continue
    }

    const info = CHECKPOINT_TYPES[g.check.type] || CHECKPOINT_TYPES.deterministic
    const color = info.color
    const gw = g.width
    const gh = g.height

    ctx.fillStyle = past ? color + '30' : (active ? color + 'dd' : color + '70')
    ctx.strokeStyle = active ? color : color + '80'
    ctx.lineWidth = active ? 2.5 : 1
    roundRect(ctx, g.x - gw / 2, g.y - gh / 2, gw, gh, 3)
    ctx.fill()
    ctx.stroke()

    if (active) {
      ctx.shadowColor = color
      ctx.shadowBlur = 14
      roundRect(ctx, g.x - gw / 2, g.y - gh / 2, gw, gh, 3)
      ctx.stroke()
      ctx.shadowBlur = 0
    }

    // Name (rotated above)
    ctx.save()
    ctx.translate(g.x, g.y - gh / 2 - 8)
    ctx.rotate(-Math.PI / 6)
    ctx.fillStyle = past ? C.dim : C.text
    ctx.font = '12px "DM Sans", system-ui, sans-serif'
    ctx.textAlign = 'left'
    ctx.fillText(g.check.name, 0, 0)
    ctx.restore()

    // Latency below
    ctx.fillStyle = C.dim
    ctx.font = '10px "JetBrains Mono", monospace'
    ctx.textAlign = 'center'
    ctx.fillText(fmtMs(g.check.latency_ms), g.x, g.y + gh / 2 + 16)

    // Hit zone for gate
    const hzPad = 6
    addHitZone(g.x - gw / 2 - hzPad, g.y - gh / 2 - hzPad, gw + hzPad * 2, gh + hzPad * 2, gateTooltip(g, info))
  }
}

function drawLLMBlock(g, active, past) {
  const bw = g.width
  const bh = g.height
  ctx.fillStyle = past ? C.prod + '35' : C.prod + '90'
  ctx.strokeStyle = C.prod
  ctx.lineWidth = active ? 2.5 : 1.5
  roundRect(ctx, g.x - bw / 2, g.y - bh / 2, bw, bh, 6)
  ctx.fill()
  ctx.stroke()

  if (active) {
    ctx.shadowColor = C.prod
    ctx.shadowBlur = 14
    roundRect(ctx, g.x - bw / 2, g.y - bh / 2, bw, bh, 6)
    ctx.stroke()
    ctx.shadowBlur = 0
  }

  ctx.fillStyle = C.bright
  ctx.font = 'bold 10px "JetBrains Mono", monospace'
  ctx.textAlign = 'center'
  ctx.fillText(g.check.name, g.x, g.y + 4)

  // Latency below (agent blocks have latency, productive LLM blocks don't)
  if (g.check.latency_ms > 0) {
    ctx.fillStyle = C.dim
    ctx.font = '10px "JetBrains Mono", monospace'
    ctx.textAlign = 'center'
    ctx.fillText(fmtMs(g.check.latency_ms), g.x, g.y + bh / 2 + 16)
  }

  // Hit zone for LLM block
  const info = g.isAgent
    ? (CHECKPOINT_TYPES[g.check.type] || { label: 'LLM Agent Call' })
    : { label: 'Productive LLM', color: C.prod }
  const hzPad = 6
  addHitZone(g.x - bw / 2 - hzPad, g.y - bh / 2 - hzPad, bw + hzPad * 2, bh + hzPad * 2, gateTooltip(g, info))
}

function gateTooltip(g, info) {
  const c = g.check
  let html = `<strong>${c.name}</strong>`
  html += `<br><span class="tt-type" style="color:${info.color || C.prod}">${info.label || c.type}</span>`
  if (c.desc) html += `<br>${c.desc}`
  if (c.latency_ms > 0) html += `<br><span class="tt-mono">${c.latency_ms}ms latency</span>`
  if (c.tokens_in > 0 || c.tokens_out > 0) {
    html += `<br><span class="tt-mono">${fmtTokens(c.tokens_in)} tok in / ${fmtTokens(c.tokens_out)} tok out</span>`
  }
  return html
}

export function drawPacket(packet) {
  for (let i = 0; i < packet.trail.length; i++) {
    const t = packet.trail[i]
    const a = Math.round((i / packet.trail.length) * 100)
    ctx.fillStyle = `rgba(56, 189, 248, ${a / 255})`
    ctx.beginPath()
    ctx.arc(t.x, t.y, 3, 0, Math.PI * 2)
    ctx.fill()
  }

  ctx.shadowColor = C.glow
  ctx.shadowBlur = 18
  ctx.fillStyle = C.glow
  ctx.beginPath()
  ctx.arc(packet.x, packet.y, 7, 0, Math.PI * 2)
  ctx.fill()
  ctx.shadowBlur = 0

  ctx.fillStyle = C.core
  ctx.beginPath()
  ctx.arc(packet.x, packet.y, 3.5, 0, Math.PI * 2)
  ctx.fill()
}

export function drawLatencyCounter(packet) {
  if (packet.state === 'done') return
  ctx.textAlign = 'center'
  ctx.fillStyle = C.bright
  ctx.font = 'bold 13px "JetBrains Mono", monospace'
  ctx.fillText(`${packet.accLatency.toFixed(0)}ms`, packet.x, packet.y + 40)
}

export function drawTitle(text) {
  const { w } = getSize()
  ctx.fillStyle = C.bright
  ctx.font = 'bold 18px "DM Sans", system-ui, sans-serif'
  ctx.textAlign = 'center'
  ctx.fillText(text, w / 2, 24)
}

export function drawLegend() {
  const { w, h } = getSize()
  let x = w * 0.04
  const y = h - 14

  ctx.font = '10px "DM Sans", system-ui, sans-serif'
  ctx.textAlign = 'left'

  // LLM Agent
  ctx.fillStyle = C.prod
  ctx.fillRect(x, y - 6, 10, 10)
  ctx.fillStyle = C.text
  ctx.fillText('LLM Agent Call', x + 14, y + 3)
  x += 110

  for (const [key, info] of Object.entries(CHECKPOINT_TYPES)) {
    if (key === 'llm' || key === 'llm_agent' || key === 'static_analysis' || key === 'crypto') continue
    ctx.fillStyle = info.color
    ctx.fillRect(x, y - 6, 10, 10)
    ctx.fillStyle = C.text
    ctx.fillText(info.label, x + 14, y + 3)
    x += ctx.measureText(info.label).width + 28
  }
}

/**
 * Draw probe annotations between baseline and drift pipelines.
 * Shows the drift process: probe entropy → synthesize rules.
 */
export function drawProbeAnnotations(pipelines, annotations) {
  if (pipelines.length < 2) return
  const baseline = pipelines[0]
  const drift = pipelines[1]

  for (const ann of annotations) {
    const srcGate = baseline.gates[ann.baselineIdx]
    const dstGate = drift.gates[ann.driftIdx]
    if (!srcGate || !dstGate) continue

    const isReplaced = ann.entropy === '0.00'
    const color = isReplaced ? '#22c55e' : '#f59e0b'

    // Draw curved arrow from baseline gate to drift gate
    const x1 = srcGate.x
    const y1 = srcGate.y + srcGate.height / 2 + 20
    const x2 = dstGate.x
    const y2 = dstGate.y - dstGate.height / 2 - 20

    const midY = (y1 + y2) / 2

    ctx.save()
    ctx.strokeStyle = color + '60'
    ctx.lineWidth = 1.5
    ctx.setLineDash([4, 3])
    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.bezierCurveTo(x1, midY, x2, midY, x2, y2)
    ctx.stroke()
    ctx.setLineDash([])

    // Arrowhead at destination
    ctx.fillStyle = color + '80'
    ctx.beginPath()
    ctx.moveTo(x2 - 4, y2 - 6)
    ctx.lineTo(x2, y2)
    ctx.lineTo(x2 + 4, y2 - 6)
    ctx.fill()
    ctx.restore()

    // Label at midpoint of the curve
    const labelX = (x1 + x2) / 2
    const labelY = midY

    // Background pill
    const label = isReplaced ? `H=0 → ${ann.result}` : `H=${ann.entropy} → keep LLM`
    ctx.font = '10px "JetBrains Mono", monospace'
    const tw = ctx.measureText(label).width
    const px = 6, py = 3
    ctx.fillStyle = C.bg + 'e0'
    roundRect(ctx, labelX - tw / 2 - px, labelY - 7 - py, tw + px * 2, 14 + py * 2, 4)
    ctx.fill()
    ctx.strokeStyle = color + '50'
    ctx.lineWidth = 1
    roundRect(ctx, labelX - tw / 2 - px, labelY - 7 - py, tw + px * 2, 14 + py * 2, 4)
    ctx.stroke()

    ctx.fillStyle = color
    ctx.textAlign = 'center'
    ctx.fillText(label, labelX, labelY + 3)

    // Hit zone for annotation
    addHitZone(
      labelX - tw / 2 - px, labelY - 7 - py,
      tw + px * 2, 14 + py * 2,
      `<strong>${ann.method}</strong><br>` +
      `Entropy: <span class="tt-mono">${ann.entropy} bits</span><br>` +
      `Result: <span style="color:${color}">${ann.result}</span><br>` +
      `<span class="tt-dim">${ann.detail}</span>`
    )
  }
}

export function drawDoneMessage(pipelines) {
  const { w, h } = getSize()

  const allDone = pipelines.every(p => p.packets[0] && p.packets[0].state === 'done')
  if (!allDone) return

  const baseline = pipelines[0].packets[0]
  const drift = pipelines[1].packets[0]

  const savedMs = baseline.accLatency - drift.accLatency
  const savedPct = ((savedMs / baseline.accLatency) * 100).toFixed(0)

  ctx.font = 'bold 14px "JetBrains Mono", monospace'
  ctx.textAlign = 'center'
  ctx.fillStyle = '#22c55e'
  ctx.fillText(
    `Drift saves ${savedMs.toFixed(0)}ms (${savedPct}%) overhead per request`,
    w / 2, h / 2
  )
}

function fmtMs(ms) {
  return `${ms}ms`
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.lineTo(x + w - r, y)
  ctx.arcTo(x + w, y, x + w, y + r, r)
  ctx.lineTo(x + w, y + h - r)
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r)
  ctx.lineTo(x + r, y + h)
  ctx.arcTo(x, y + h, x, y + h - r, r)
  ctx.lineTo(x, y + r)
  ctx.arcTo(x, y, x + r, y, r)
  ctx.closePath()
}
