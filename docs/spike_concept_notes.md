# Spike Concept Note
**Date:** Day 9
**Sources read:**
- Wikipedia: Leaky Integrate-and-Fire model
- SpikingJelly Encoding documentation


## Q1. What is a spike in a biological neuron?

A biological neuron receives electrical input from thousands of 
other neurons. Its membrane voltage accumulates — like water filling 
a leaky bucket. When voltage crosses a threshold, the neuron fires 
a sharp electrical pulse (the spike) and resets. The "leaky" part 
means unused charge slowly dissipates — recent inputs matter more 
than old ones.


## Q2. What is a spike in our context?

In our INR/BRL data, most daily returns are tiny — near zero.
Occasionally, a large price move occurs (COVID crash, rate decision).
We treat these large return events as "spikes" — rare, high-information 
moments in an otherwise quiet signal. This maps naturally onto biological 
spike coding: silence = no signal, burst = important event.


## Q3. What is rate coding?

Rate coding encodes signal strength as the number of spikes 
per unit time. A high-return day produces many spikes in the 
encoding window; a flat day produces few or none. The model 
reads spike count as signal intensity — simple, robust, 
and easy to train.



## Q4. What is temporal coding?

Starter to rewrite:
Temporal coding encodes information in the exact timing of spikes, 
not their count. A strong input causes a spike to fire earlier in 
the time window; a weak input fires later. This is more expressive 
than rate coding but requires the network to be sensitive to 
millisecond-level timing — fragile for noisy financial data.

---

## Q5. Which will we use and why?

We will use rate coding (specifically SpikingJelly's PoissonEncoder).
Financial return data is noisy — exact spike timing would be 
dominated by noise rather than signal. Rate coding averages over 
a time window, making it robust to individual noisy observations.
It is also more interpretable: spike rate ≈ signal strength, 
which maps cleanly onto "how large was this price move."
Finally, rate coding networks are more stable to train with 
surrogate gradient methods, which we'll use in Month 2.

---

## Personal observations after reading:

I still have doubt like how this data is transformed using 
this binary codes like 0 ad 1 .

---

## One question I still have:

 "If LIF neurons reset after every spike, how do they 
 remember anything about the previous timestep?"
```

---

### 🧠 The core mental model to lock in today
```
Biology               →    Our FX System
─────────────────────────────────────────
Membrane voltage      →    Accumulated return signal
Threshold             →    "This move is large enough to matter"
Spike fires           →    Encode this timestep as 1 (active)
No spike              →    Encode as 0 (quiet)
Leaky decay           →    Recent prices weighted more than old
Rate = spikes/window  →    Strength of the current price signal
