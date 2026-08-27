import { describe, expect, it } from 'vitest'

import { transposeEffect } from '../components/textInput.js'

// Ctrl+T is readline's transpose-chars, matching bash:
//   - cursor at EOL: swap the last two characters, cursor stays at EOL
//   - cursor mid-line: swap the character under the cursor with the one
//     before it, then advance the cursor one cluster
//   - cursor at the start / fewer than two clusters: no-op

describe('transposeEffect (Ctrl+T transpose-chars)', () => {
  it('swaps the last two characters when the cursor is at EOL', () => {
    // "hell|o" -> swap l and o -> "helol", cursor stays at EOL.
    expect(transposeEffect('hello', 5)).toEqual({ value: 'helol', cursor: 5 })
  })

  it('swaps the character under the cursor with the previous one, mid-line', () => {
    // "ab|cd": cursor on 'c' -> swap b and c -> "acbd", cursor advances past it.
    expect(transposeEffect('abcd', 2)).toEqual({ value: 'acbd', cursor: 3 })
  })

  it('swaps the last two clusters mid-line and leaves the remainder untouched', () => {
    // "xy|z": cursor on 'z' -> swap y and z -> "xzy", cursor advances to EOL.
    expect(transposeEffect('xyz', 2)).toEqual({ value: 'xzy', cursor: 3 })
  })

  it('swaps the first two characters when the cursor is on the second', () => {
    // "|ab": cursor on 'b' -> swap a and b -> "ba", cursor at EOL.
    expect(transposeEffect('ab', 1)).toEqual({ value: 'ba', cursor: 2 })
  })

  it('is a no-op when the cursor is at the start of the line', () => {
    expect(transposeEffect('ab', 0)).toEqual({ value: 'ab', cursor: 0 })
  })

  it('is a no-op with fewer than two clusters', () => {
    expect(transposeEffect('', 0)).toEqual({ value: '', cursor: 0 })
    expect(transposeEffect('a', 1)).toEqual({ value: 'a', cursor: 1 })
  })

  it('keeps multi-byte grapheme clusters intact (emoji, EOL)', () => {
    const value = 'a\ud83d\ude00b' // a 😀 b (emoji = one cluster, 2 UTF-16 units)
    // The last two clusters are [b] and [😀]; swap them, cursor stays at EOL.
    expect(transposeEffect(value, value.length)).toEqual({ value: 'ab\ud83d\ude00', cursor: 4 })
  })

  it('keeps multi-byte grapheme clusters intact (emoji, mid-line)', () => {
    const value = 'a\ud83d\ude00b' // a 😀 b
    // Cursor on the emoji (snaps to its start, UTF-16 index 1): swap it with
    // 'a' -> "😀ab", cursor lands past the emoji.
    expect(transposeEffect(value, 1)).toEqual({ value: '\ud83d\ude00ab', cursor: 3 })
  })

  it('keeps combining sequences intact (e + combining acute)', () => {
    const eAcute = '\u0065\u0301' // e + U+0301 combining acute = one cluster
    const value = 'ab' + eAcute
    // EOL: swap 'b' and the e-acute cluster: a<e-acute>b, cursor at EOL.
    expect(transposeEffect(value, value.length)).toEqual({
      value: 'a' + eAcute + 'b',
      cursor: value.length
    })
  })

  it('keeps combining sequences intact mid-line', () => {
    const eAcute = '\u0065\u0301'
    const value = 'ab' + eAcute
    // Cursor on the e-acute cluster: swap it with 'b' -> a<e-acute>b, cursor
    // lands past the cluster (EOL here).
    expect(transposeEffect(value, 2)).toEqual({
      value: 'a' + eAcute + 'b',
      cursor: value.length
    })
  })
})
