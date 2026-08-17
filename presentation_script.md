# Presentation Script

## Slide 1 – Title

Good morning everyone. I'm Shivam, roll number CS25M048. My project is on DL-SIFA — applying deep learning to perform statistical ineffective fault analysis on AES-128. Let me walk you through it.

---

## Slide 2 – Outline

So this is the outline that I'm going to cover in my presentation.

---

## Slide 3 – Background

In fault attacks, the attacker physically disturbs the device during encryption — using a voltage glitch, laser pulse, or electromagnetic interference. This causes wrong computations, and the faulty ciphertext leaks information about the secret key. This is the basis of Differential Fault Analysis, or DFA.

To defend against this, implementations use temporal redundancy. The device encrypts the same plaintext twice and compares both results. If the results don't match, it means a fault occurred, so the output is suppressed. This effectively blocks DFA since the attacker never receives the faulty ciphertext.

---

## Slide 4 – SIFA

However, this defense has a weakness. When a fault is injected, roughly 50% of the time the targeted bit was already in the faulted state. For example, if we are forcing a bit to 0 and it was already 0, then the fault changes nothing. The device encrypts twice, both outputs match, and the ciphertext is released normally.

The issue is that these released ciphertexts are not uniformly random. They carry a statistical bias introduced by the fault constraint. SIFA exploits this bias to recover the key — using only the correct outputs.

For this project, I implemented the attack in two ways — first using the traditional mathematical approach, and second using a neural network. Then I compared how many traces each method requires for key recovery.

---

## Slide 5 – Fault Model

Here is my experimental setup. I inject a stuck-at-zero fault on bit 0 of one byte, at the input to Round 8 SubBytes.

The reason this creates a bias is straightforward. Normally, the byte at the fault point can take any value from 0 to 255 — all 256 values are equally likely. But after the stuck-at-zero fault, only the cases where bit 0 was already zero will pass the redundancy check. That limits us to 128 values — only even numbers. So if we build a histogram of these byte values, half the bins will be empty. That is the statistical signal we exploit.

I ran 50,000 fault attempts per byte position and collected approximately 25,000 ineffective traces for each, which matches the expected 50% rate.

---

## Slide 6 – Traditional SIFA

The traditional approach is straightforward. We take one byte of the last round key K10 and try all 256 possible values. For each guess, we invert the ciphertexts back through rounds 10, 9, and 8 to reach the fault point.

Then we check what fraction of traces have bit 0 equal to zero. If the guess is correct, this fraction will be exactly 1.0, because in all ineffective fault traces that bit was already zero. If the guess is wrong, we get a random distribution, so the fraction is approximately 0.5.

In my results, the correct key byte is 0xD0 and it was recovered successfully. All 16 byte positions show perfect bias of 1.0. The minimum number of traces required for recovery is 20.

---

## Slide 7 – DL-SIFA

Now, the limitation of the traditional method is that it requires exact knowledge of the fault model — which bit was faulted, the fault type, and the injection round. In practical hardware attacks, this information is often unavailable or imprecise.

So I used a neural network instead. Rather than checking a specific bit, the network looks at the entire 256-bin histogram of byte values at the fault point.

For profiling, I used 500 random keys and collected both faulted and clean histograms for each. This gave me 3,000 training samples. The network architecture is a multi-layer perceptron — 256 inputs going through three hidden layers down to a single sigmoid output. I trained it for 200 epochs using Adam optimizer with binary cross-entropy loss.

During the attack, I follow the same 256-guess approach — for each key guess, I invert, build a histogram, and score it with the trained network. The guess with the highest score is the recovered key.

---

## Slide 8 – DL-SIFA Results

Training accuracy reached 100%. The correct key byte 0xD0 receives a score of 1.0, while all 255 wrong guesses score close to 0. Key recovery succeeds with 50 traces.

One notable observation — the separation between correct and wrong guesses is actually stronger than the traditional method. Traditional gives 1.0 versus 0.5; the neural network gives 1.0 versus 0.0. So while it requires slightly more traces, the confidence in the answer is higher.

---

## Slide 9 – Comparison

Comparing both methods side by side — both successfully recover the correct key byte. The traditional approach needs only 20 traces and is computationally simple since it checks a single bit. The neural network needs 50 traces because it requires enough samples for the histogram pattern to be clear.

However, the neural network does not require any prior knowledge of the fault model. It learns the bias pattern from profiling data. This makes it more suitable for real-world scenarios where the fault characteristics are uncertain or noisy.

---

## Slide 10 – Results Plot

These four plots summarize the results visually. Top-left shows perfect bias across all 16 bytes. Top-right shows the traditional key distinguisher — the correct guess at bias 1.0 and all wrong guesses at 0.5. Bottom-left shows the neural network distinguisher — correct guess at 1.0, everything else near zero. And bottom-right shows the progressive trace comparison — traditional succeeds at 20 traces, DL-SIFA at 50.

---

## Slide 11 – Conclusion

To conclude — SIFA is able to break AES-128 with temporal redundancy by exploiting only the correct outputs from ineffective faults. Both the traditional and neural network approaches successfully recover the key.

The traditional method is faster and simpler when the fault model is known precisely. The DL-SIFA approach requires more traces but does not depend on knowing the fault model, which is the more realistic scenario in practice.

As future work, it would be valuable to test with noisy and imprecise fault models to quantify the robustness advantage of the neural network approach.

---

## Slide 12 – Thank You

That concludes my presentation. Thank you for your attention. I'm happy to take any questions.
