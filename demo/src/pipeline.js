/**
 * Pipeline animation engine.
 *
 * Manages packets flowing through security checkpoints.
 * Each packet is a glowing dot that travels left-to-right,
 * pausing at each gate proportional to its latency.
 *
 * Supports multiple lanes (laneY) for side-by-side comparison.
 */

import { CHECKPOINT_TYPES, flattenChecks } from './models.js'

// Animation-time scale: how fast gates process (higher = faster animation)
const SPEED_SCALE = 0.1125

// Layout constants (fractions of canvas width/height)
const LEFT_PAD = 0.04
const RIGHT_PAD = 0.96
const GATE_W = 8
const GATE_H = 36

/** Pipeline state for one model. laneY is a fraction of canvas height (e.g. 0.35 or 0.70). */
export function createPipeline(model, steps, laneY) {
  const checksPerStep = flattenChecks(model)
  return {
    model,
    steps,
    laneY: laneY || 0.50,
    checksPerStep,
    gates: [],
    packets: [],
    totalChecks: checksPerStep.length * steps,
    totalLatency: checksPerStep.reduce((s, c) => s + c.latency_ms, 0) * steps,
    totalTokens: checksPerStep.reduce((s, c) => s + (c.tokens_in || 0) + (c.tokens_out || 0), 0) * steps,
    completedPackets: 0,
  }
}

/** Compute gate positions across the full pipeline (all steps). */
export function layoutGates(pipeline, w, h) {
  const { checksPerStep, steps, laneY } = pipeline
  const gates = []

  const inputChecks = checksPerStep.filter(c => c.phase === 'input')
  const outputChecks = checksPerStep.filter(c => c.phase !== 'input')

  // Only insert the auto "LLM" block if the model has a productive LLM phase
  // (i.e. safety-tax style). For drift-style models where LLM calls are in the
  // checks themselves, skip it.
  const hasLLMChecks = checksPerStep.some(c => c.type === 'llm_agent' || c.type === 'llm')
  const hasProductiveLLM = !hasLLMChecks  // safety-tax models have no llm_agent checks

  const slotsPerStep = inputChecks.length + (hasProductiveLLM ? 1 : 0) + outputChecks.length
  const totalSlots = slotsPerStep * steps
  const usableW = (RIGHT_PAD - LEFT_PAD) * w
  const spacing = usableW / (totalSlots + 1)
  const startX = LEFT_PAD * w
  const y = laneY * h

  let slotIdx = 0
  for (let s = 0; s < steps; s++) {
    for (const c of inputChecks) {
      slotIdx++
      const isAgent = c.type === 'llm_agent' || c.type === 'llm'
      gates.push({
        x: startX + slotIdx * spacing,
        y, check: c, step: s,
        isLLM: false,
        isAgent,  // renders as blue block but still pauses
        width: isAgent ? GATE_W * 2.5 : GATE_W,
        height: isAgent ? GATE_H * 1.3 : GATE_H,
      })
    }
    if (hasProductiveLLM) {
      slotIdx++
      gates.push({
        x: startX + slotIdx * spacing,
        y,
        check: { name: steps > 1 ? `LLM ${s + 1}` : 'LLM', type: 'productive', latency_ms: 0, desc: 'Productive inference' },
        step: s, isLLM: true, isAgent: false,
        width: GATE_W * 2.5, height: GATE_H * 1.3,
      })
    }
    for (const c of outputChecks) {
      slotIdx++
      gates.push({
        x: startX + slotIdx * spacing,
        y, check: c, step: s,
        isLLM: false, isAgent: false,
        width: GATE_W, height: GATE_H,
      })
    }
  }

  pipeline.gates = gates
  return gates
}

/** Spawn a new packet at the left edge. */
export function spawnPacket(pipeline, w, h) {
  pipeline.packets.push({
    x: LEFT_PAD * w - 20,
    y: pipeline.laneY * h,
    targetGateIdx: 0,
    waitFrames: 0,
    state: 'moving',
    trail: [],
    accLatency: 0,
    accTokensIn: 0,
    accTokensOut: 0,
    currentCheckName: '',
    speed: 3,
  })
}

/** Advance all packets one frame. Returns true if any packet is still active. */
export function tickPackets(pipeline, dt) {
  let anyActive = false

  for (const p of pipeline.packets) {
    if (p.state === 'done') continue
    anyActive = true

    if (p.state === 'waiting') {
      p.waitFrames -= dt
      if (p.waitFrames <= 0) {
        p.state = 'moving'
        p.targetGateIdx++
      }
      continue
    }

    if (p.targetGateIdx >= pipeline.gates.length) {
      p.x += p.speed * dt * 1.5
      const rightEdge = pipeline.gates[pipeline.gates.length - 1].x + 50
      if (p.x > rightEdge) {
        p.state = 'done'
        pipeline.completedPackets++
      }
      p.trail.push({ x: p.x, y: p.y })
      if (p.trail.length > 30) p.trail.shift()
      continue
    }

    const gate = pipeline.gates[p.targetGateIdx]
    const dx = gate.x - p.x

    if (Math.abs(dx) < p.speed * dt + 1) {
      p.x = gate.x
      if (gate.isLLM) {
        // Skip productive LLM blocks (zero latency, just visual)
        p.targetGateIdx++
      } else {
        p.state = 'waiting'
        p.waitFrames = gate.check.latency_ms * SPEED_SCALE
        p.accLatency += gate.check.latency_ms
        p.accTokensIn += gate.check.tokens_in || 0
        p.accTokensOut += gate.check.tokens_out || 0
        p.currentCheckName = gate.check.name
      }
    } else {
      p.x += Math.sign(dx) * p.speed * dt
    }

    p.trail.push({ x: p.x, y: p.y })
    if (p.trail.length > 30) p.trail.shift()
  }

  return anyActive
}

/** Snap the first active packet to its current target gate (for pause). */
export function snapToCurrentGate(pipeline) {
  for (const p of pipeline.packets) {
    if (p.state === 'done') continue

    if (p.state === 'waiting') {
      p.waitFrames = 0
      break
    }

    while (p.targetGateIdx < pipeline.gates.length && pipeline.gates[p.targetGateIdx].isLLM) {
      p.targetGateIdx++
    }

    if (p.targetGateIdx < pipeline.gates.length) {
      const gate = pipeline.gates[p.targetGateIdx]
      p.x = gate.x
      p.trail = [{ x: p.x, y: p.y }]
      p.state = 'waiting'
      p.waitFrames = 0
      p.accLatency += gate.check.latency_ms
      p.accTokensIn += gate.check.tokens_in || 0
      p.accTokensOut += gate.check.tokens_out || 0
      p.currentCheckName = gate.check.name
    } else {
      p.x = pipeline.gates[pipeline.gates.length - 1].x + 50
      p.trail = [{ x: p.x, y: p.y }]
      p.state = 'done'
      pipeline.completedPackets++
    }
    break
  }
}

/** Jump the first active packet to the next gate instantly. */
export function stepToNextGate(pipeline) {
  for (const p of pipeline.packets) {
    if (p.state === 'done') continue

    if (p.state === 'waiting') {
      p.waitFrames = 0
      p.state = 'moving'
      p.targetGateIdx++
    }

    while (p.targetGateIdx < pipeline.gates.length && pipeline.gates[p.targetGateIdx].isLLM) {
      p.targetGateIdx++
    }

    if (p.targetGateIdx < pipeline.gates.length) {
      const gate = pipeline.gates[p.targetGateIdx]
      p.x = gate.x
      p.trail = [{ x: p.x, y: p.y }]
      p.state = 'waiting'
      p.waitFrames = 0
      p.accLatency += gate.check.latency_ms
      p.accTokensIn += gate.check.tokens_in || 0
      p.accTokensOut += gate.check.tokens_out || 0
      p.currentCheckName = gate.check.name
    } else {
      p.x = pipeline.gates[pipeline.gates.length - 1].x + 50
      p.trail = [{ x: p.x, y: p.y }]
      p.state = 'done'
      pipeline.completedPackets++
    }

    break
  }
}
