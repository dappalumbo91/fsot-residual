# Direction: compress with FSOT, restore from the compressed state

The restore is almost perfect when the decoder already holds the law that generated the message, and the compressed state still carries every residual that law does not fix. The shared law is the FSOT 2.1 engine. The compressed state is an index into that engine plus a residual. The release reads the index back and adds the residual.

That is the whole method. The sections below say why the public theory allows it, how the unifying concept names the same pieces, what the earlier compressor actually measured, and the first contract to run.

## 1. The compressed state

A message \(X\) is restored by a decoder that already holds side information \(Y\). \(Y\) is the engine: the five seeds, \(K\), the domain phase, \(D_{\mathrm{eff}}\), the gauge library, and the folds those already determine.

Slepian and Wolf (1973) proved that if the decoder holds \(Y\) and the encoder does not, a lossless code still achieves the conditional entropy \(H(X \mid Y)\). The rate is the uncertainty left after the library is applied. Wyner and Ziv (1976) give the matching lossy statement: the rate to hold distortion under \(D\) is \(I(X; Z \mid Y)\), which is smaller than the rate with an empty decoder.

So the file that moves is short when the library is rich, and the restore is exact when the residual is kept.

\[
\text{compressed state} = (\text{phase},\; D_{\mathrm{eff}},\; \text{gauge index},\; \text{residual})
\]

- **Phase** says whether the message is free flow or a bound condensate, and which fold the observation reads. The unifying concept puts this first: the same medium, read at a different fold, is a different object.
- **\(D_{\mathrm{eff}}\)** is the zoom. Wilson's renormalization group keeps the relevant operators at a scale and integrates the rest out. The natural zoom of a message is the smallest integer \(D_{\mathrm{eff}}\) at which the residual falls inside the declared floor. A lower zoom has thrown away a relevant piece. A higher zoom is storing pieces the floor already calls irrelevant.
- **Gauge index** is the library address: the domain, the leaf, the codon home, the K-line place. This is the part the white-hole stroke releases.
- **Residual** is \(X\) minus what that address fixes. Keeping it is lossless relative to the floor. Dropping it is lossy, and the distortion is then the score.

POOF is the condenser stroke: write the index, and pay for every bit the residual no longer carries. SUCTION is the release stroke: the decoder draws the pattern back out of the library. In the engine, POOF \(\approx 0.153482\), SUCTION \(\approx 0.147034\), and POOF/(POOF+SUCTION) \(\approx 0.510729\). `docs/APPLY_THERMO.md` already scores that fraction against \(1/2\) as a literature band. It is the heat/information pump. It is a pump fraction. The compression ratio of a particular file is a measured rate, computed after the encode, from \(H(X \mid Y)\) or from the byte length of the residual.

\(\beta\) stays as the engine defines it. Raising it so \(T_3\) fills a gap would move every domain scalar. The correspondence check fails if \(|T_3|\) exceeds \(10^{-15}\). The compressed state uses \(T_1\) and \(T_2\) the way the working catalog does.

## 2. What "almost perfect" is

Write the floor before the run. Score in the object's own unit.

| Object | Floor | Almost perfect means |
| --- | --- | --- |
| Bit string whose law is in the library | Hamming distance 0 | Every bit returns |
| Symbol on a settled gauge (amino acid, kind, integer identity) | Lattice drop, DROP \(\approx 0.238\), with the gauge inside \(10^{-4}\) of its home | The name returns |
| Scalar already on a published bar | \(z = \|value - center\| / \sigma \le 1\), \(\sigma\) the published one-sided uncertainty | The value sits inside that bar |
| Message with a declared distortion | Rate-distortion \(D\) written in the contract | The average distortion is \(\le D\) |

A fair coin has entropy 1 bit. Shannon's source coding theorem (1948) says a lossless code of an arbitrary source needs a rate at least the entropy. A code that restores an incompressible control from fewer than \(H(X)\) bits has used a hidden copy of \(X\), or it has miscounted the rate. Kolmogorov complexity is the shortest program that emits \(X\). It is uncomputable in general, so this project uses the engine as a declared model class. The model class is the MDL description of Rissanen (1978): store the law, then store the bits the law does not predict. The law here is fixed. It is not refit per file.

A degenerate code restores the object it was written for. The genetic code maps 64 codons onto 20 amino acids and stop. The amino acid can return exactly. The original codon does not, unless the residual still names which codon it was. Four codons share serine's complement pairs. Sign-flipped codon sums are one magnitude with two assignments. The compressor has to say which of those it is returning.

Quantum messages add two hard limits. Wootters and Zurek (1982): a single unknown quantum state cannot be cloned. Schumacher (1995): a quantum source compresses to its von Neumann entropy, and Holevo's bound caps the classical bits you can read back out. A quantum restore is a decode of a code onto a fresh system. Copying the original system is a different operation, and the no-cloning theorem forbids it for an unknown state.

## 3. The unifying concept, read as information flow

The desktop document *FSOT - Unified Concept* is the physical picture. Each of its pieces is a compression object.

| Concept in the document | Information object | Public anchor |
| --- | --- | --- |
| Depth holds a library | Decoder side information \(Y\) | Slepian–Wolf, Wyner–Ziv |
| Zoom level | \(D_{\mathrm{eff}}\), relevant operators only | Wilson, renormalization group |
| Black hole condenses and passes through depth | Encoder, horizon encoding | Bekenstein bound; Hawking radiation; Page curve |
| White hole releases | Decoder | Page (1993): the information returns over the whole radiation, after the halfway point |
| POOF and SUCTION | The two strokes of one pump | Landauer (1961): erasing a bit costs \(kT\ln 2\). Bennett (1982): the cost is paid at erasure, which is what lets Maxwell's demon balance |
| Viscosity | Channel noise | Shannon channel coding; polar codes (Arıkan 2009) spend redundancy to cross a noisy channel |
| Tension holding a droplet | Minimum distance of a code | Hamming, Kraft inequality: a prefix code satisfies \(\sum 2^{-\ell_i} \le 1\) |
| Phase (steam, liquid, ice) | Which codebook applies | The same bits, read in the wrong phase, are a different message |
| Light is the thin ripple, sound the thick one | A nearly noiseless channel versus a thick one | Capacity \(C = \tfrac12 \ln(1 + S/N)\) for the Gaussian channel grows as the noise drops |
| Coherence versus emergence | Settled gauge versus new residual | A system inside the floor is stable. A residual above the floor is new information |
| Observation settles the outcome | Which fold is read | Measurement chooses the codebook. It is a discrete fold of the engine, with `observed`, hits, and \(\delta\psi\) already defined there |
| As above, so below | One code at every zoom | The same \((\mathrm{phase}, D_{\mathrm{eff}}, \mathrm{index}, \mathrm{residual})\) describes an atom, a codon, a storm, or a file |
| Life organizes energy flow | A persistent pattern with throughput | A whirlpool is a short description plus a continuing current. The description is the compressed state. The current is the channel use |
| Consciousness as divergence and coalescence | One viewpoint versus a system-wide code | Tishby's information bottleneck (1999) keeps the bits about \(Y\) that still predict \(X\). A viewpoint that keeps every bit has not compressed. A viewpoint that drops the predictive bits cannot restore |

The Page curve is the warning on the release stroke. Early Hawking radiation looks thermal. The information is in the correlations that show up over the whole emission, as made precise by the island formula (Penington 2019; Almheiri, Engelhardt, Marolf, Maxfield 2019). Reading only the gauge index, and skipping the residual, is looking at the early radiation. The pattern is there. The original message is still in the part you skipped.

Holography ('t Hooft 1993; Susskind 1995) says a region's information can sit on its boundary, with entropy scaling as area. The compressed state is that boundary record. The bulk message is what the release rebuilds. FSOT's own note on observer-patch holography (`scripts/build_oph_fsot_challenge_panel.py`) keeps the seeds as the primary object and the observer as a coupling fold. This project follows that order. The library is the seed engine. The observer fold selects which address is read. A screen of patches is a picture of the boundary record.

## 4. Domains, and what each one contributes

The same four-tuple covers these. Each domain tells you which failure to expect.

**Classical bits.** Huffman (1952), arithmetic coding, and asymmetric numeral systems (Duda, arXiv:0902.0271, the coder inside modern LZMA-class compressors) approach the entropy of a known model. Lempel–Ziv (1977) builds the model as it reads. Use them as the baseline on the control file. FSOT replaces the adaptive model only where the engine already contains the law. On prose, images, and genomes that have not been reduced to a gauge, Lempel–Ziv is the honest compressor, and FSOT has nothing to add until a relation is extracted. The lattice rule already says this: raw text, FASTQ, and images are not teachable until a relation exists.

**Sparse signals.** Candès, Romberg, and Tao (2006) and Donoho (2006) proved that a vector with \(s\) nonzeros in a known basis is recovered exactly from about \(s \log(n/s)\) linear measurements, when the sensing matrix satisfies the restricted isometry property. The basis has to be the true one. A sparse code in the wrong basis fails. The FSOT analogue of the basis is the gauge library. A message that is one gauge plus a small residual is the sparse case. A message that is uniform noise is the dense case, and compressed sensing correctly refuses it.

**Noisy channels.** Capacity is the most mutual information across the channel. Below capacity, codes exist whose block error goes to zero (Shannon). Polar codes and LDPC codes are constructions that approach it. Viscosity in the unifying concept is this channel. Tension is the redundancy you must leave in the compressed state if the restore has to cross a thick medium. A frictionless channel (the document's light) can carry a residual with almost no extra bits. A thick channel (the document's sound) cannot.

**Rate distortion.** Shannon (1959) defined \(R(D)\), the smallest rate that holds average distortion at \(D\). "Almost" is a \(D\). For a Gaussian source under squared error, \(R(D) = \tfrac12 \log(\sigma^2 / D)\) for \(D \le \sigma^2\). Halving the error costs half a bit per sample. There is no free almost-perfect. The fertile-window fidelity near 0.80 in the 2.0 simulation is a point on some \(R(D)\) curve. Turning it into a smaller \(D\) costs residual bits, which have to be counted in the rate.

**Physics of the black hole.** Bekenstein (1973) bounded the entropy of a region. Hawking (1975) made the information paradox: thermal radiation seems to destroy the infalling state. Page (1993) computed that the entanglement entropy of the radiation turns over and comes back down. The condenser does not delete the message. It relocates it. A compressor that cannot invert has modeled the paradox, not the Page curve.

**Thermodynamics.** Landauer's \(kT \ln 2\) is the heat of erasure. Bennett showed that computation itself can be reversible, so the heat appears when information is discarded. A lossless FSOT code erases nothing and owes no Landauer heat beyond the ordinary work of writing the residual. A lossy code owes \(kT \ln 2\) per erased bit. POOF is the name of that stroke in the engine. Equating the pump fraction 0.511 to a file's compression ratio would be a new coefficient. Leave the fraction on its literature band.

**Learning and minds.** The information bottleneck compresses a representation until only the bits that predict the target remain. Predictive coding (Rao and Ballard 1999; Friston's free-energy review 2010) stores a prediction and passes the error. That error is the residual in the four-tuple. Language models are compressors of their training text (Delétang et al., arXiv:2309.10668). They are a baseline on prose. They are a fitted model with a large parameter count. This project keeps the fixed engine and does not graft a language model in as the library.

**Biology.** The codon table is the worked example of a many-to-one code, already chewed on the lattice at pin AEB2AD: codon onto amino acid lands as a shared label, and the sign-flipped sums stay one magnitude. Compression succeeds at the amino-acid zoom. Recompilation of the base string requires the residual trit. Protein folding is the same shape one zoom up: many sequences, one fold, and the sequence does not come back from the fold alone.

**Renormalization and scale.** A coarse description is a legal compression. Restoring the fine description needs the integrated-out pieces, which are the residual. The unifying concept's "as above, so below" says the form of the four-tuple repeats. It also says the contents of the residual change with zoom. A galaxy-scale index does not recompile a nucleotide.

## 5. What is already measured

**2.0 Intelligence Compressor** (`Intelligence_Compression_Domain.md`, 2026-06-17). Optimal point recorded there: \(D_{\mathrm{eff}} = 12\), observed, \(\delta\psi = 0.8\), recent hits 1, \(S \approx 0.2736\), and a continuous `attention_weight` scaling the observer factor. Decoder fidelity is described as about 0.80 after phase optimization. That file is a simulation note. `attention_weight` is a continuous dial, so those parameters are not a 2.1 encode.

**Lean intelligence-compression benchmark** (2026-08-05, authority path the AEB2AD engine, residual domain Neuroscience). 87 records, pooled median error 0.018003%. Fertile-slice median error 20.378%. Optimal \(S\) about 0.27346 against a fertile-center target 0.2736. `best_intelligence_score` about 0.9997 is the distance of that score to 1. The 0.018% is the scalar residual of `make_fsot_record` scaled in Neuroscience. A message was not put in and taken back out.

**Information-theory public panel.** 21 records, \(D_{\mathrm{eff}}\) stored as 8 in the benchmark file, median error 0 at display precision, eval kind literature anchor. The anchors include \(\ln 2 \approx 0.693147\), fair-coin entropy 1, Kraft bound 1, a channel-capacity proxy 1.618, and a Gaussian-channel proxy 0.4812. Computed equals the stored anchor. The panel is the list of objects any later derivation has to meet. A derivation that pastes the anchor back is the measurement copied down.

**Lattice, pin AEB2AD.** Settled gauges restore names inside DROP \(\approx 0.238\), with true remainders under about \(10^{-4}\) and false remainders near \(K \approx 0.420\). Codon to amino acid, codon to trit triple, and the integer identities are restores of symbols whose homes are in the library. They are the nearest existing proof that an index plus a tiny residual recompiles a discrete object. They do not recompile a raw byte string.

**C++ standing, 2026-10-07.** The port matches the Python authority. The precision gate is \(z \le 1\) on 91 record rows. Held-out confirmations are 0. Host evaluations (neutral mesons, \(g_A\), radius, \(\Delta{-}N\), neutron moment, atmospheric angle, CP phase) are later readings on the same pin. Use the port when a compressor inner loop needs speed. Cite a host value only for the object that header scores.

## 6. First contract

Three files, one floor each, scored once. No new coefficient. Pin AEB2AD. The engine file is read from the Lean tree and is not edited.

**Contract A. Degenerate code.** Object 1: the amino acid of each of the 64 codons. Object 2: the codon string itself. Library: the settled codon-to-amino-acid gauges already kept beside the lattice (`kept_codon_amino.json` and the English names), read at pin AEB2AD. Floor for object 1: exact label, residual under \(10^{-4}\). Floor for object 2: exact three-letter restore. Expected physics: object 1 meets the floor from the index alone; object 2 meets the floor only when the residual names the synonymous codon. A run that restores the bases without storing that residual has hidden the codon.

**Contract B. Entropy control.** Two strings of equal length. String Law is produced by a named identity the engine already evaluates (a short digit sequence of a seed expression, with the expression's name stored as the index). String Control is incompressible: a fixed public string whose compressed length under a standard ANS or zlib coder sits at its own length. Floor: Hamming distance 0 on both restores. Rate: bytes of the compressed state, index and residual both counted. Expected physics: Law's rate is the index plus a short residual and sits under Control's rate. Control's rate sits at the standard coder's rate. A Law rate of zero means the residual was left in the decoder by hand and must be added back into the count.

**Contract C. Declared distortion.** Same String Law. Drop every residual component smaller than a floor \(D\) chosen in advance and written here before the run: \(D = 10^{-4}\) in the gauge unit used by Contract A. Score the reconstruction error in that unit, and score the bits erased. Expected physics: the error is \(\le D\), and the erased-bit count times \(\ln 2\) is the Landauer ledger of this stroke. The pump fraction stays 0.510729. It is not adjusted to match the erasure count.

A contract that fails stays failed. The next move is to name which piece of the four-tuple was wrong (phase, zoom, index, or residual), then write a new contract. Fitting a multiplier so the restore lands closer is a failed contract with an extra step.

## 7. Reading list

The directions in sections 2–4 come from these. Dates and titles are the standard citations.

- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal* 27, 379–423 and 623–656. Source coding and channel coding.
- Shannon, C. E. (1959). Coding theorems for a discrete source with a fidelity criterion. Rate distortion \(R(D)\).
- Kraft, L. G. (1949). Prefix-code inequality. Huffman, D. A. (1952). Minimum-redundancy codes. Ziv, J. and Lempel, A. (1977). Universal sequential compression.
- Kolmogorov, A. N. (1965). Three approaches to the quantitative definition of information. *Problems of Information Transmission* 1, 1–7.
- Rissanen, J. (1978). Modeling by shortest data description. *Automatica* 14, 465–471.
- Slepian, D. and Wolf, J. K. (1973). Noiseless coding of correlated information sources. *IEEE Trans. Inf. Theory* 19, 471–480.
- Wyner, A. D. and Ziv, J. (1976). The rate-distortion function for source coding with side information at the decoder. *IEEE Trans. Inf. Theory* 22, 1–10.
- Cover, T. M. and Thomas, J. A. *Elements of Information Theory*. Typical sets, AEP, the Gaussian channel.
- MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge.
- Landauer, R. (1961). Irreversibility and heat generation in the computing process. *IBM J. Res. Dev.* 5, 183–191.
- Bennett, C. H. (1982). The thermodynamics of computation. *Int. J. Theor. Phys.* 21, 905–940.
- Bekenstein, J. D. (1973). Black holes and entropy. *Phys. Rev. D* 7, 2333.
- Hawking, S. W. (1975). Particle creation by black holes. *Commun. Math. Phys.* 43, 199.
- Page, D. N. (1993). Information in black hole radiation. *Phys. Rev. Lett.* 71, 3743.
- Penington, G. (2019). Entanglement wedge reconstruction and the information paradox. arXiv:1905.08255. Almheiri, A., Engelhardt, N., Marolf, D., and Maxfield, H. (2019). The entropy of bulk quantum fields and the entanglement wedge of an evaporating black hole. arXiv:1905.08762.
- 't Hooft, G. (1993). Dimensional reduction in quantum gravity. Susskind, L. (1995). The world as a hologram. *J. Math. Phys.* 36, 6377.
- Wilson, K. G. (1971). Renormalization group and critical phenomena. *Phys. Rev. B* 4, 3174.
- Candès, E. J., Romberg, J., and Tao, T. (2006). Robust uncertainty principles: exact signal reconstruction from highly incomplete frequency information. *IEEE Trans. Inf. Theory* 52, 489–509. Donoho, D. L. (2006). Compressed sensing. *IEEE Trans. Inf. Theory* 52, 1289–1306.
- Wootters, W. K. and Zurek, W. H. (1982). A single quantum cannot be cloned. *Nature* 299, 802–803.
- Schumacher, B. (1995). Quantum coding. *Phys. Rev. A* 51, 2738. Holevo, A. S. (1973). Bounds for the quantity of information transmitted by a quantum communication channel. *Problems of Information Transmission* 9, 177.
- Arıkan, E. (2009). Channel polarization: a method for constructing capacity-achieving codes. *IEEE Trans. Inf. Theory* 55, 3051–3073.
- Tishby, N., Pereira, F. C., and Bialek, W. (1999). The information bottleneck method. arXiv:physics/0004057.
- Rao, R. P. N. and Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience* 2, 79–87.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience* 11, 127–138.
- Delétang, G. et al. (2023). Language modeling is compression. arXiv:2309.10668.
- Duda, J. (2009). Asymmetric numeral systems. arXiv:0902.0271.
- Crick, F. H. C. The genetic code is degenerate: 64 codons, 20 amino acids and stop. The lattice tables in FSOT Intelligence are the local scored form of that map, pin AEB2AD.

Local FSOT sources for the same map: *FSOT - Unified Concept.docx*; Lean `docs/APPLY_THERMO.md`, `docs/BH_WH_CLAIM_EVIDENCE.md`, `docs/FSOT_CORRESPONDENCE.md`; C++ `README.md` current standing and `include/fsot/core.hpp`.

## 8. Next build, when a contract is run

A small encoder and decoder in this folder. They import the engine from `C:\Users\damia\Desktop\FSOT-2.1-Lean\vendor\fsot_compute.py` and, if the inner loop needs the port, call the C++ host for the same pin. They write three artifacts: the compressed state, the restored object, and a score against the floor in section 6. They leave the hubs untouched.

The first run is Contract A, because the library for it already exists and the two objects (amino acid, codon) teach the degenerate-code limit on data this project already trusts.
