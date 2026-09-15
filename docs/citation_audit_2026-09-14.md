# Citation audit — main.bib vs. main.tex + sec/*.tex

Generated 2026-09-14. Sources: Crossref REST API (`api.crossref.org/works/<DOI>`), arXiv export API, Semantic Scholar.
Scope: every key cited in main.tex and sec/1..7 (mdpi_full_version/ ignored). 77 distinct cited keys; main.bib holds 77 real entries (+30 `@String` macros).
Every cited key resolves to a bib entry and every bib entry is cited (see section C).

Verdicts: OK = matches source; FIX = source clearly contradicts entry; UNVERIFIED = could not confirm from an authoritative source.

## A. Per-entry table

| key | verdict | what is wrong | corrected BibTeX fields |
|---|---|---|---|
| reza2004realization | OK | matches Crossref (J. VLSI Signal Process. 38(1):35-44, 2004) | — |
| thomas2010distance | OK | matches Crossref (J. Appl. Ecol. 47(1):5-14, 2010) | — |
| buckland2023wildlife | FIX | `pages={1--22}` is not the journal pagination; Crossref gives article number 20, no page range | `pages = {20},` (article number) |
| royle2004nmixture | OK | matches Crossref (Biometrics 60(1):108-115) | — |
| royle2009bayesian | OK | matches Crossref (Ecology 90(11):3233-3244) | — |
| iijima2020review | OK | matches Crossref (Mammal Study 45(3):177-, 2020) | — |
| morgan2024wildlife | OK | matches Crossref (PLOS ONE 19(10):e0310020) | — |
| burke2019optimizing | OK | matches Crossref (IJRS 40(2):439-467, 2019) | — |
| karp2020detecting | OK | correct but incomplete: Sci. Rep. 10(1), article 5220 | optional: `number = {1}, pages = {5220},` |
| ulhaq2021automated | FIX | first author is published as "Anwaar Ulhaq" (one word) in Crossref and on the MDPI page; bib has "Ul Haq, Anwaar" (also "Ul Haq et al." / "Ul Haq 2021" in sec/2_related.tex lines 32-33 and 215) | `author = {Ulhaq, Anwaar and Adams, Peter and Cox, Tarnya E. and Khan, Asim and Low, Tom and Paul, Manoranjan},` |
| mowen2022improving | OK | matches Crossref (Sustainability 14(19):12133) | — |
| hyun2020remotely | OK | matches Crossref (Animals 10(12):2387) | — |
| mcmahon2021comparing | OK | matches Crossref; note vol. 49(1):54-65 is the Feb-2022 print issue (online 15 Sep 2021) — `year={2022}` would match the volume, 2021 is defensible as online-first | optional: `year = {2022},` |
| nazir2021advances | OK | matches Crossref (Ecol. Inform. 61:101212, 2021) | — |
| prosekov2020methods | OK | matches Crossref (Forests 11(8):808) | — |
| brown2022automated | OK | matches Crossref (Comput. Electron. Agric. 193:106689, 2022) | — |
| infantes2022automated | OK | correct but incomplete: Front. Ecol. Evol. 10, article 905309 | optional: `pages = {905309},` |
| mpouziotas2023automated | OK | matches Crossref (Appl. Sci. 13(13):7787) | — |
| kilfoil2020using | OK | matches Crossref (ICES J. Mar. Sci. 77(7-8):2882-2889) | — |
| tarling2022deep | OK | matches Crossref (PLOS ONE 17(5):e0267759) | — |
| oviedo2024automatic | OK | matches Crossref (IJAAA 11(4), 2024) | — |
| singh2020animal | FIX | **wrong DOI**: 10.1109/WACV45572.2020.9093549 resolves to Machiraju & Balasubramanian, "A Little Fog for a Large Turn" (WACV 2020, pp. 2891-2900). Title/authors/pages of the entry are correct for WACV 2020 pp. 1427-1438 | `doi = {10.1109/WACV45572.2020.9093504},` |
| beery2020synthetic | OK | matches Crossref (WACV 2020, pp. 852-862) | — |
| liu2023lote | OK | matches Crossref (ICCV 2023, pp. 20007-20018) | — |
| zhu2021tph | OK | matches Crossref (ICCVW 2021, pp. 2778-2788) | — |
| sapkota2025yoloreview | FIX | author list truncated: the paper has 12 authors (Crossref/OpenAlex); bib keeps 4 and silently drops 8 middle authors. Journal/vol/issue/year OK | `author = {Sapkota, Ranjan and Flores-Calero, Marco and Qureshi, Rizwan and Badgujar, Chetan and Nepal, Upesh and Poulose, Alwin and Zeno, Peter and Vaddevolu, Uday Bhanu Prakash and Khan, Sheheryar and Shoman, Maged and Yan, Hong and Karkee, Manoj},` |
| zhang2023motrv2 | OK | matches Crossref (CVPR 2023, pp. 22056-22065) | — |
| dolokov2023upper | OK | matches Crossref (VISAPP 2023 / VISIGRAPP proceedings, pp. 945-952) | — |
| miele2020revisiting | FIX | **title wrong**: published title is "Revisiting animal photo-identification using deep **metric** learning and network analysis" (bib omits "metric"). MEE 12(5):863-873, 2021 otherwise OK | `title = {Revisiting animal photo-identification using deep metric learning and network analysis},` |
| li2024redeformtr | OK | matches Crossref (IEEE Access 12:106321-106332) | — |
| nguyen2023trustworthy | OK | matches Crossref (TPAMI 45(7):8538-8552) | — |
| oksuz2022lrp | OK | matches Crossref (TPAMI 44(12):9446-9463) | — |
| cai2021cascade | OK | matches Crossref (TPAMI 43(5):1483-1498) | — |
| hidayatullah2025yolov8 | FIX | arXiv:2501.13400 has since been published (DataCite IsVersionOf → Crossref): Jurnal RESTI 10(2):341-354, 21 Apr 2026, DOI 10.29207/resti.v10i2.6598, under the revised title "YOLOv8 to YOLO11 Performance Benchmark and Comprehensive Architectural Comparative Review". Also 3rd author is "Muhammad Rizqi Sholahuddin" (bib: "Sholahuddin, M.") | see full entry in B (cite the journal version) |
| valmadre2021local | OK | arXiv-only (OpenAlex: no published version); metadata matches | — |
| cheng2021boundary | OK | note: IEEE/Crossref paginate the DOI'd version 15329-15337; bib's 15334-15342 is the CVF open-access pagination — both legitimate | optional (to match the DOI): `pages = {15329--15337},` |
| rahman2021perframe | OK | matches Crossref (WACVW 2021, pp. 152-160) | — |
| vercauteren2011managing | OK | note: Crossref/T&F ebook record paginates the chapter 514-549; bib's 501-535 is the print pagination that the literature cites. Entry lacks the editor | optional: `editor = {Hewitt, David G.}, address = {Boca Raton, FL},` |
| jennelle2018applying | FIX | **author list wrong**: 4th author Erik E. Osnas is missing (paper has 11 authors, bib has 10); "Demarest, D." and "Gubler, R." are truncated (E. David Demarest, Rolf Gubler); "Rolley, Robert E." → Crossref "Robert Rolley" | `author = {Jennelle, Christopher S. and Walsh, Daniel P. and Samuel, Michael D. and Osnas, Erik E. and Rolley, Robert E. and Langenberg, Julia and Powers, Jenny G. and Monello, Ryan J. and Demarest, E. David and Gubler, Rolf and Heisey, Dennis M.},` |
| pickering2022divergent | FIX | paper has 40 authors; bib silently stops after the 10th (no `and others`), so the reference reads as a 10-author paper | `author = {Pickering, Bradley and Lung, Oliver and Maguire, Finlay and Kruczkiewicz, Peter and Kotwa, Jonathon D. and Buchanan, Tore and Gagnier, Marianne and Guthrie, Jennifer L. and Jardine, Claire M. and Marchand-Austin, Alex and others},` |
| kay2022caltech | OK | matches Crossref (ECCV 2022, LNCS, pp. 290-311) | — |
| shuai2021confluence | FIX | **4th author missing**: Crossref lists Shepley, Falzon, Kwan, **Brankovic**. Journal/vol/pages/year OK (TPAMI 45(10):11561-11574, 2023). (Key says "shuai2021" but that is only a key) | `author = {Shepley, Andrew J. and Falzon, Greg and Kwan, Paul and Brankovic, Ljiljana},` |
| bodla2017softnms | OK | matches Crossref (ICCV 2017, pp. 5562-5570) | — |
| diefenbach2025accounting | OK | matches Crossref (J. Appl. Ecol. 62(4):986-994, 2025) | — |
| everingham2010pascal | OK | matches Crossref (IJCV 88(2):303-338, 2010) | — |
| zadrozny2002transforming | OK | matches Crossref (KDD 2002, pp. 694-699) | — |
| naeini2015obtaining | OK | matches Crossref (Proc. AAAI 29(1), 2015) | — |
| brier1950verification | OK | matches Crossref (Mon. Wea. Rev. 78(1):1-3) | — |
| wilson1927probable | OK | matches Crossref (JASA 22(158):209-212) | — |
| hong2013poisson | OK | matches Crossref (CSDA 59:41-51) | — |
| hartley2004multiple | OK | matches Crossref (CUP, 2nd ed., 2004) | — |
| buckland2015distance | OK | matches Crossref (Springer, Methods in Statistical Ecology, 2015) | — |
| redmon2016yolo | OK | matches Crossref via 10.1109/CVPR.2016.91 (CVPR 2016, pp. 779-788); no DOI/pages in bib | optional: `pages = {779--788}, doi = {10.1109/CVPR.2016.91},` |
| zhang2016single | OK | matches Crossref (CVPR 2016, pp. 589-597) | optional: `pages = {589--597}, doi = {10.1109/CVPR.2016.70},` |
| zhang2020atss | OK | matches Crossref (CVPR 2020, pp. 9756-9765) | optional: `pages = {9756--9765}, doi = {10.1109/CVPR42600.2020.00978},` |
| feng2021tood | OK | matches Crossref (ICCV 2021, pp. 3490-3499) | optional: `pages = {3490--3499}, doi = {10.1109/ICCV48922.2021.00349},` |
| bewley2016sort | OK | matches Crossref (ICIP 2016, pp. 3464-3468) | optional: `pages = {3464--3468}, doi = {10.1109/ICIP.2016.7533003},` |
| wojke2017deepsort | OK | matches Crossref (ICIP 2017, pp. 3645-3649) | optional: `pages = {3645--3649}, doi = {10.1109/ICIP.2017.8296962},` |
| carion2020detr | OK | matches Crossref (ECCV 2020, LNCS, pp. 213-229) | optional: `pages = {213--229}, doi = {10.1007/978-3-030-58452-8_13},` |
| lin2014coco | OK | matches Crossref (ECCV 2014, LNCS, pp. 740-755) | optional: `pages = {740--755}, doi = {10.1007/978-3-319-10602-1_48},` |
| beery2018recognition | OK | matches Crossref (ECCV 2018, LNCS, pp. 472-489) | optional: `pages = {472--489}, doi = {10.1007/978-3-030-01270-0_28},` |
| zhang2022bytetrack | OK | matches Crossref (ECCV 2022, LNCS, pp. 1-21) | optional: `pages = {1--21}, doi = {10.1007/978-3-031-20047-2_1},` |
| wang2024yolov9 | OK | matches Crossref (ECCV 2024, LNCS, pp. 1-21; the LNCS volume carries a 2025 imprint, conference year 2024 is standard) | optional: `pages = {1--21}, doi = {10.1007/978-3-031-72751-1_1},` |
| zhao2024rtdetr | OK | matches Crossref (CVPR 2024, pp. 16965-16974) | optional: `pages = {16965--16974}, doi = {10.1109/CVPR52733.2024.01605},` |
| norouzzadeh2018automatically | OK | matches Crossref (PNAS 115(25), 2018); no DOI in bib | optional: `doi = {10.1073/pnas.1719367115},` |
| kellenberger2018detecting | OK | matches Crossref (RSE 216:139-153, 2018); no DOI in bib | optional: `doi = {10.1016/j.rse.2018.06.028},` |
| corcoran2019automated | FIX | **entry is a chimera of two papers**: the title ("Automated detection of wildlife using drones: Synthesis, opportunities and constraints") belongs to Corcoran, Winsen, Sudholz & Hamilton, MEE 12(6):1103-1114, **2021**, DOI 10.1111/2041-210X.13581; the author list (Corcoran, Denman, Hanger, Wilson, Hamilton) and year 2019 belong to "Automated detection of koalas using low-level aerial surveillance and machine learning", Sci. Rep. 9:3208, DOI 10.1038/s41598-019-39917-5. `volume={10}, number={11}, pages={1874--1887}` match neither. The only citation (table row "Corcoran 2019 / Thermal / UAV / Koala") describes the koala paper | see B (koala paper; alternative MEE-review entry also given) |
| lyu2022rtmdet | OK | arXiv:2212.07784, DataCite: no published version; authors/title match | — |
| aharon2022botsort | OK | arXiv:2206.14651, DataCite: no published version; authors/title match | — |
| kline2024integrating | FIX | **title truncated**: DataCite/OpenAlex title is "Integrating Biological Data into Autonomous Remote Sensing Systems for In Situ Imageomics: A Case Study for Kenyan Animal Behavior Sensing with Unmanned Aerial Vehicles (UAVs)". arXiv-only (no published version found) | `title = {Integrating Biological Data into Autonomous Remote Sensing Systems for In Situ Imageomics: A Case Study for {Kenyan} Animal Behavior Sensing with Unmanned Aerial Vehicles ({UAVs})},` |
| adam2025wildlifereid10k | FIX | arXiv:2406.09211 (2024) has since been published: CVPR Workshops 2025, pp. 2090-2100, DOI 10.1109/CVPRW67362.2025.00197 (Crossref). Bib cites the preprint with year 2024 while the table label reads "Adam 2025" | see B (cite the CVPRW 2025 version) |
| wang2021nwd | OK | arXiv:2110.13389, DataCite: no published version of this title; authors match | — |
| ren2015faster | OK | NeurIPS 28 (2015), pp. 91-99 (OpenAlex); a TPAMI 39(6):1137-1149 (2017) version also exists — citing NeurIPS is fine | optional: `pages = {91--99},` |
| wang2024yolov10 | OK | NeurIPS 37 (2024) confirmed (OpenAlex, DOI 10.52202/079017-3429) | — |
| zhang2023dino | OK | title/authors match arXiv:2203.03605; ICLR 2023 venue (OpenReview id 3mRwyG5one) could not be machine-confirmed in this run (OpenReview/DBLP/S2 blocked) but is well established | — |
| guo2017calibration | OK | matches PMLR: Proc. 34th ICML, PMLR 70:1321-1330, 2017 | — |
| vanrijsbergen1979information | OK | 2nd ed., Butterworths, London, 1979 confirmed via the contemporaneous JASIS review record (DOI 10.1002/asi.4630300621) | — |

## B. Full corrected BibTeX entries (FIX items)

```bibtex
@article{buckland2023wildlife,
  author = {Buckland, S. T. and Borchers, D. L. and Marques, T. A. and Fewster, R. M.},
  title = {Wildlife Population Assessment: Changing Priorities Driven by Technological Advances},
  journal = {Journal of Statistical Theory and Practice}, volume = {17}, number = {2}, pages = {20}, year = {2023},
  doi = {10.1007/s42519-023-00319-6}}

@article{ulhaq2021automated,
  author = {Ulhaq, Anwaar and Adams, Peter and Cox, Tarnya E. and Khan, Asim and Low, Tom and Paul, Manoranjan},
  title = {Automated Detection of Animals in Low-Resolution Airborne Thermal Imagery},
  journal = {Remote Sensing}, volume = {13}, number = {16}, pages = {3276}, year = {2021}, doi = {10.3390/rs13163276}}

@inproceedings{singh2020animal,
  author = {Singh, Abhineet and Pietrasik, Marcin and Natha, Gabriell and Ghouaiel, Nehla and Brizel, Ken and Ray, Nilanjan},
  title = {Animal Detection in Man-made Environments},
  booktitle = {IEEE Winter Conf. Appl. Comput. Vis. (WACV)}, pages = {1427--1438}, year = {2020},
  doi = {10.1109/WACV45572.2020.9093504}}

@article{sapkota2025yoloreview,
  author = {Sapkota, Ranjan and Flores-Calero, Marco and Qureshi, Rizwan and Badgujar, Chetan and Nepal, Upesh and Poulose, Alwin and Zeno, Peter and Vaddevolu, Uday Bhanu Prakash and Khan, Sheheryar and Shoman, Maged and Yan, Hong and Karkee, Manoj},
  title = {YOLO advances to its genesis: a decadal and comprehensive review of the You Only Look Once (YOLO) series},
  journal = {Artificial Intelligence Review}, volume = {58}, number = {9}, pages = {274}, year = {2025},
  doi = {10.1007/s10462-025-11253-3}}

@article{miele2020revisiting,
  author = {Miele, Vincent and Dussert, Gaspard and Spataro, Bruno and Chamaill\'{e}-Jammes, Simon and Allain\'{e}, Dominique and Bonenfant, Christophe},
  title = {Revisiting animal photo-identification using deep metric learning and network analysis},
  journal = {Methods in Ecology and Evolution}, volume = {12}, number = {5}, pages = {863--873}, year = {2021},
  doi = {10.1111/2041-210X.13577}}

% published version of arXiv:2501.13400 (key kept so \cite{} calls need no change)
@article{hidayatullah2025yolov8,
  author = {Hidayatullah, Priyanto and Syakrani, Nurjannah and Sholahuddin, Muhammad Rizqi and Gelar, Trisna and Tubagus, Refdinal},
  title = {{YOLOv8} to {YOLO11} Performance Benchmark and Comprehensive Architectural Comparative Review},
  journal = {Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi)}, volume = {10}, number = {2}, pages = {341--354}, year = {2026},
  doi = {10.29207/resti.v10i2.6598}}

@article{jennelle2018applying,
  author = {Jennelle, Christopher S. and Walsh, Daniel P. and Samuel, Michael D. and Osnas, Erik E. and Rolley, Robert E. and Langenberg, Julia and Powers, Jenny G. and Monello, Ryan J. and Demarest, E. David and Gubler, Rolf and Heisey, Dennis M.},
  title = {Applying a Bayesian weighted surveillance approach to detect chronic wasting disease in white-tailed deer},
  journal = {Journal of Applied Ecology}, volume = {55}, number = {6}, pages = {2944--2953}, year = {2018},
  doi = {10.1111/1365-2664.13178}}

@article{pickering2022divergent,
  author = {Pickering, Bradley and Lung, Oliver and Maguire, Finlay and Kruczkiewicz, Peter and Kotwa, Jonathon D. and Buchanan, Tore and Gagnier, Marianne and Guthrie, Jennifer L. and Jardine, Claire M. and Marchand-Austin, Alex and others},
  title = {Divergent {SARS-CoV-2} variant emerges in white-tailed deer with deer-to-human transmission},
  journal = {Nature Microbiology}, volume = {7}, number = {12}, pages = {2011--2024}, year = {2022},
  doi = {10.1038/s41564-022-01268-9}}

@article{shuai2021confluence,
  author = {Shepley, Andrew J. and Falzon, Greg and Kwan, Paul and Brankovic, Ljiljana},
  title = {Confluence: A Robust Non-{IoU} Alternative to Non-Maxima Suppression in Object Detection},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume = {45}, number = {10}, pages = {11561--11574}, year = {2023},
  doi = {10.1109/TPAMI.2023.3273210}}

% the paper the table row (Thermal / UAV / Koala, 2019) actually describes
@article{corcoran2019automated,
  author = {Corcoran, Evangeline and Denman, Simon and Hanger, Jon and Wilson, Bree and Hamilton, Grant},
  title = {Automated detection of koalas using low-level aerial surveillance and machine learning},
  journal = {Scientific Reports}, volume = {9}, number = {1}, pages = {3208}, year = {2019},
  doi = {10.1038/s41598-019-39917-5}}
% only if the drone review was intended instead (then the table row's year/taxa must change):
@article{corcoran2021drones,
  author = {Corcoran, Evangeline and Winsen, Megan and Sudholz, Ashlee and Hamilton, Grant},
  title = {Automated detection of wildlife using drones: Synthesis, opportunities and constraints},
  journal = {Methods in Ecology and Evolution}, volume = {12}, number = {6}, pages = {1103--1114}, year = {2021},
  doi = {10.1111/2041-210X.13581}}

@article{kline2024integrating,
  author = {Kline, Jenna M. and Kholiavchenko, Maksim and Brookes, Otto and Berger-Wolf, Tanya and Stewart, Charles V. and Stewart, Christopher},
  title = {Integrating Biological Data into Autonomous Remote Sensing Systems for In Situ Imageomics: A Case Study for {Kenyan} Animal Behavior Sensing with Unmanned Aerial Vehicles ({UAVs})},
  journal = {arXiv preprint arXiv:2407.16864}, year = {2024}, doi = {10.48550/arxiv.2407.16864}}

% published version of arXiv:2406.09211
@inproceedings{adam2025wildlifereid10k,
  author = {Adam, Luk\'{a}\v{s} and \v{C}erm\'{a}k, Vojt\v{e}ch and Papafitsoros, Kostas and Picek, Lukas},
  title = {{WildlifeReID-10k}: Wildlife Re-Identification Dataset with 10k Individual Animals},
  booktitle = {IEEE/CVF Conf. Comput. Vis. Pattern Recog. Worksh. (CVPRW)}, pages = {2090--2100}, year = {2025},
  doi = {10.1109/CVPRW67362.2025.00197}}
```

## C. Bib entries not cited anywhere

None. All 77 entries in main.bib are cited in main.tex/sec/*.tex, and all 77 cited keys resolve (no duplicate keys, no two keys for the same work).

## D. Claim-support flags

1. **wang2021nwd — numbers misattributed** (sec/2_related.tex:93-95):
   > "Wang et al.~\cite{wang2021nwd} show that IoU is extremely sensitive to localisation error at small scale: a one-pixel shift on a $6\times6$ box drops IoU from 0.53 to 0.06, while the same shift on a $36\times36$ box drops it from 0.90 to 0.65."

   In the NWD paper (Fig. 1) 0.53 and 0.90 are the IoUs *after* a one-pixel diagonal shift of the 6x6 and 36x36 boxes; 0.06 and 0.65 are after a *four*-pixel shift. Geometry confirms it: 6x6 shifted 1 px: 25/47 = 0.53, 4 px: 4/68 = 0.06; 36x36 shifted 1 px: 1225/1367 = 0.90, 4 px: 1024/1568 = 0.65. Suggested wording: "a one-pixel diagonal shift drops the IoU of a 6x6 box to 0.53 and a four-pixel shift to 0.06, whereas a 36x36 box keeps 0.90 and 0.65 under the same shifts." (The "6.7 AP" gain is as stated in the NWD abstract.)

2. **corcoran2019automated — wrong paper behind the citation** (sec/2_related.tex:217, table row "Corcoran 2019 & Thermal & UAV & Koala"): the bib entry's title is the 2021 MEE drone review, which is neither a 2019 paper nor a koala study. The row matches the 2019 Sci. Rep. koala paper; use the corrected entry in B.

3. **adam2025wildlifereid10k — label/year mismatch** (sec/2_related.tex:224 "Adam 2025"): bib year is 2024 (arXiv). Citing the CVPRW 2025 version (B) resolves it.

4. **ulhaq2021automated — author name in text** (sec/2_related.tex:32-33 "Ul Haq et al.", :215 "Ul Haq 2021"): should read "Ulhaq" to match the published byline.

Soft flags (wording only, no clear contradiction — fix at the authors' discretion):
- sec/2_related.tex:107-109: "Query-based detectors, DETR~\cite{carion2020detr} and DINO~\cite{zhang2023dino}, converge slowly and struggle on very small objects, which RT-DETR~\cite{zhao2024rtdetr} targets directly." DINO's own headline contribution is *faster* convergence than DETR (denoising training, 12-epoch results), and RT-DETR targets real-time/NMS-free inference rather than small objects. Consider attributing slow convergence to DETR only and describing RT-DETR as targeting real-time inference.
- sec/2_related.tex:195-196: "Modern detectors are systematically overconfident~\cite{guo2017calibration}". Guo et al. study image *classifiers*; "modern neural networks" would match the source.

All other citations were checked against the verified titles/abstracts and are plausibly supported (e.g. sapkota2025 does cover v1-v12; hidayatullah2025 does note missing publications for several releases; kay2022caltech does report that oracle detections remove most counting error; dolokov2023 does exploit a known upper bound on animal count).

## E. Totals

- Checked: 77 cited keys (all cited keys; all bib entries).
- OK: 65 (of which 20 carry optional completeness suggestions — missing DOI/pages/article number, editor, or a pagination/year convention note).
- FIX: 12 — buckland2023wildlife, ulhaq2021automated, singh2020animal, sapkota2025yoloreview, miele2020revisiting, hidayatullah2025yolov8, jennelle2018applying, pickering2022divergent, shuai2021confluence, corcoran2019automated, kline2024integrating, adam2025wildlifereid10k.
- UNVERIFIED: 0 (zhang2023dino's ICLR-2023 venue is well established but could not be confirmed by API during this run; its title/authors were verified).
- Retractions: none found (Crossref records carry no retraction/update notices for any checked DOI).
- Most consequential: singh2020animal (DOI points to an unrelated paper), corcoran2019automated (chimera entry), miele2020revisiting (wrong title), shuai2021confluence / jennelle2018applying / sapkota2025yoloreview (missing co-authors), and the wang2021nwd figure numbers in the text.
