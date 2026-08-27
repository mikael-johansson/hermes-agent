import { describe, expect, it } from 'vitest'

import { transposeEffect } from '../components/textInput.js'

// Ctrl+T is readline's transpose-chars: swap the two grapheme clusters
// immediately before the cursor and move the cursor one cluster to the right
// (clamped at EOL). The common "I mistyped" case is the cursor at the end of
// the line, where it swaps the last two characters.

describe('transposeEffect (Ctrl+T transpose-chars)', () => {
  it('swaps the last two characters when the cursor is at EOL', () => {
    // "hell|o" -> swap l and o -> "helol"
    expect(transposeEffect('hello', 5)).toEqual({ value: 'helol', cursor: 5 })
  })

  it('swaps mid-line and moves the cursor one cluster right', () => {
    // "ab|cd": cursor at 2, swaps a and b -> "ba|cd", cursor advances to 3.
    expect(transposeEffect('abcd', 2)).toEqual({ value: 'bacd', cursor: 3 })
  })

  it('is a no-op when the cursor is at the start of the line', () => {
    expect(transposeEffect('ab', 0)).toEqual({ value: 'ab', cursor: 0 })
  })

  it('is a no-op with fewer than two clusters', () => {
    expect(transposeEffect('', 0)).toEqual({ value: '', cursor: 0 })
    expect(transposeEffect('a', 1)).toEqual({ value: 'a', cursor: 1 })
  })

  it('keeps multi-byte grapheme clusters intact (emoji)', () => {
    const value = 'a\ud83d\ude00b' // a 😀 b (emoji = one cluster, 2 UTF-16 units)
    // The two clusters before EOL are [😀] and [b]; swap them, cursor stays at EOL.
    expect(transposeEffect(value, value.length)).toEqual({ value: 'ab\ud83d\ude00', cursor: 4 })
  })

  it('keeps combining sequences intact (e + combining acute)', () => {
    const eAcute = '\u0065\u0301' // e + U+0301 combining acute = one cluster
    const value = 'ab' + eAcute
    // Swap 'b' and the e-acute cluster: a<e-acute>b, cursor at end.
    expect(transposeEffect(value, value.length)).toEqual({
      value: 'a' + eAcute + 'b',
      cursor: value.length
    })
  })

  it('swaps two clusters mid-line and leaves the remainder untouched', () => {
    // Cursor after the second cluster of "x y z": swap x and y -> "y x z".
    expect(transposeEffect('xyz', 2)).toEqual({ value: 'yxz', cursor: 3 })
  })
})
