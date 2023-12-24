explicit_stage_statement_examples = """
PATHOLOGIC STAGING: pT1a(m) NX.
The pTNM code should read: pT1c NX based on the size of this tumor.
TNM CODE: pT2 pNO MX G3 of 3.
STAGING: pT1b pNX
pTNM: pT2 NO MX G3.
pTNM pT2 NX MX Grade 3 of 3.
pTNM CODE: T1b NO G1 of 3
pTNM: T1b Nx Mx
pTNM: T1b NO MX
pTNM: pT1c NO MX.
STAGING CODE: pTib(m) NO G3
pTMN: pT1b NO MX.
pTNM CODE: pT1c NX MX G 3 of 3.
STAGING CODE pTic NO G3 of 3.
PATHOLOGIC STAGING: pT1b NX
pT1c NO Mx
TNM STAGING: pT2(m) pN1a
STAGING: pT1c pNO(sn).
Primary tumor:pT2
Pathologic staging: T1c N3 MX
PATHOLOGICAL STAGE: PT1B PNX.
STAGING CODE: PT1c NO G2 of 3.
pTNM CODE: pT2 NO G3 of 3
pTNM CODE: T2 NO G3 of 3
pTNM CODE: T2 NO G3 OF 3.
STAGING: pT2 pNO(sn)
Pathologic staging: pT2 NO Mx
STAGING CODE: pT2 NO
Pathological stage: pT2 pNO pMx G2 (Nottingham score 6)
STAGING: pT2 pNO(sn)
pTNM CODE: pT2 NO G3 of 3
pTNM CODE: T2 NO G3 of 3
"""

t_measurement_statement_examples = """
Invasive carcinoma size: 1.4 cm
TUMOR SIZE: 3.5 cm. maximally,
TUMOR SIZE: 3.8.cm
TUMOR SIZE: GREATEST DIMENSION OF LARGEST FOCUS OF INVASION: 5 CM
TUMOR SIZE: 2.2 cm maximally.
TUMOR SIZE: 2.5 x 2.4 x 1.6 cm.
TUMOR SIZE: 2.5 cm,
TUMOR SIZE: 4.5 X 1.5 X 2.7 CM.
TUMOR SIZE 1.5mm.
TUMOR SIZE: 0.8 CM
"""

t_lookup = {
# "Primary tumor cannot be assessed",
"Tx": {
"Notes": """
Primary tumor cannot be assessed
Report would explicity state that the tumor cannot be assessed.
""",
"Relevant Vocabulary": """
Not assessed
No detectable tumor
Undetectable
No detectable tumors
Absence of detectable tumor
No assessable tumor detected
""",
"Sample Sentences": """
Tumor assessment not possible in this specimen
Tumor not assessable in this specimen
Not possible to assess tumor
Inadequate specimen for tumor assessment
Sample does not permit tumor assessment

"""
},
"T0": {
"Notes": """
No evidence of primary tumor
Report would either explicity say no evidence, or would make no remarks on the tumor at all.
""",
"Relevant Vocabulary": """
detectable
evidence
measurable
No
Absence
Undetectable
""",
"Sample Sentences": """
No tumor detectable (or no detectable tumor)
No evidence of tumor
Tumor undetectable
No measurable tumor
Absence of detectable tumor
Absence of tumor
"""
},
# "Ductal carcinoma in situ",
"Tis (DCIS)": {
"Notes": """
Ductal carcinoma in situ"
""",
"Relevant Vocabulary": """
DCIS
Ductal
Situ
""",
"Sample Sentences": """
DCIS: PRESENT, HIGH GRADE, EXTENSIVE.
DCIS: Present, intermediate grade - not extensive.
IN-SITU COMPONENT: DCIS present,high grade with comedonecrosis, negative for extensive intraductal component (EIC)
DCIS: Present, high-grade, not extensive.DISTANCE TO CLOSEST MARGIN: DCIS extends to within 1 high power field of the inked margin of resection; invasive carcinoma extends to within 2.0 mm of the inked margin of resection.
DISTANCE OF DCIS FROM NEAREST\nMARGIN: Within 1 mm of superior margin.
DCIS: Present, intermediate grade, not extensive.
MICROCALCIFICATIONS: PRESENT IN DCIS
DUCTAL CARCINOMA IN SITU,NOT OTHERWISE SPECIFIED,NOTTINGHAM COMBINED GRADE 3
DCIS: Present, intermediated nuclear grade, not extensive.
LEFT BREAST MASS, EXCISION WITH NEEDLE LOCALIZATION: INVASIVE DUCTAL ADENOCARCINOMA, - DUCTAL CARCINOMA IN-SITU
DCIS: Present, high-grade, not extensive.
DIFFERENTIATED DUCTAL CARCINOMA IN SITU, TOTAL NOTTINGHAM SCORE OF 8
PROBABLY GREATER THAN 1.0 CM FOR DCIS.
DCIS: PRESENT, HIGH GRADE, NOT EXTENSIVE.
CARCINOMA IN-SITU, HIGH GRADE, SOLID AND COMEDO TYPE IS PRESENT ADJACENT TO THE INVASIVE CARCINOMA AND REPRESENTS APPROXIMATELY 5% OF THE TUMOR MASS
DCIS: Present, not extensive.
DUCTAL IN-SITU COMPONENT: DCIS present, nuclear grade 2 (intemediate), solid and cribiform patterns
Breast (89 g), left, lumpectomy: -Invasive ductalcarcinoma -Grade (Modified Bloom Richardson score): 3 of 3. -Tubule score:3 of 3 -Nuclear score: 3 of 3 -Mitotic score: 3 of3 -Margins: -Invasive carcinoma measures < 1mm from the anterior surgical margin
THERE IS A RESIDUAL FOCUS OF DCIS.
DCIS: DCIS present, solid and cribriform with focal central comedonecrosis, nuclear grade 2 (intermediate).
SIZE OF DCIS: Estimated size of 8 mm, present in a multiple foci
DCIS: PRESENT, HIGH NUCLEAR GRADE, NOT EXTENSIVE.
DISTANCE OF DCIS FROM NEAREST\nMARGIN: Within 1 mm of superior margin.
DCIS DISTANCE TO MARGIN: DCIS approaches to 2 mm from lateral margin.
Keep in mind that there is also a LOBULAR carcinoma in situ
The key words are IN SITU or IS
DCIS
LCIS
""",

"Negative Sample Sentences": """
DUCTAL CARCINOMA IN SITU: None identified.
DUCTAL CARCINOMA IN SITU
DCIS: None seen.
DUCTAL CARCINOMA IN SITU None seen
UNINVOLVED BY DUCTAL CARCINOMA IN-SITU
DCIS: MARGINS UNINVOLVED BY DCIS: DISTANCE OF DCIS FROM CLOSEST MARGIN: SEE COMMENT.
DCIS: None detected
LCIS: None detected
"""
},
# "Paget disease not associated with invasive carcinoma or DCIS",
"Tis (Paget)": {
"Notes": """
Paget disease not associated with invasive carcinoma or DCIS
""",


"Relevant Vocabulary": """
Paget
Nipple
Areola
""",
"Sample Sentences": """
Paget's disease of the breast
Breast Paget's disease
Paget's disease present
Nipple carcinoma present
Nipple involvement
Areola involvement
""",
"Negative Sample Sentences": """
No areola involvement
No nipple involvement
No Paget's disease detected 
Paget's disease absent (or undetectable)
Nipple uninvolved
Areola uninvolved
""",
},
# "Largest Tumor size <= 1 mm",
"T1mi": {
"Notes": """
Largest Tumor size <= 1 mm
""",


"Relevant Vocabulary": """
1 mm
Largest diameter
Diameter
""",
"Sample Sentences": """
Largest diameter < 1 mm
Less than 1 mm in largest diameter
Largest measurable diameter < 1 mm (also 1mm)
Major diameter less than 1 mm (also 1mm)
"""
},
# "Largest Tumor size > 1 mm but <= 5 mm",
"T1a": {
"Notes": """
Largest Tumor size > 1 mm but <= 5 mm
""",


"Relevant Vocabulary": """
Largest diameter (or major diameter) larger than 1mm (1mm)
Largest diameter (or major diameter) >1 mm (or 1mm)
""",
"Sample Sentences": """
RIGHT BREAST - LUMPECTOMY SPECIMEN - MODERATELY DIFFERENTIATED INFILTRATING DUCTAL CARCINOMA.
SIZE OF INVASIVE COMPONENT: 1.1m.
TUMOR FOCALITY: Unifocal.
LYMPHOVASCULAR INVASION: None seen.
PERINEURAL INVASION: None seen.
SURGICAL MARGINS: Free of disease.
The specimen consists a mass of yellow-tan to pink fibrofatty tissue measuring 5.1 x 3x 1.6. cm.
Serially sectioning reveals an ill-defined tan firm mass measuring approximately 1.1 cm in its greatest dimensions and located approximately 0.2 cm from the nearest margin of resection.
INFILTRATING DUCT CARCINOMA
TUMOR SIZE Less than 5 millimeters
LYMPHOVASCULAR INVASION None seen
SURGICAL MARGINS Uninvolved by carcinoma
NEAREST MARGIN Inferior margin at 0.5 cm
HISTOLOGIC TYPE: Invasive mammary carcinoma of no special type with mucinous features. [this is fine but it is unrelated to staging. Invasive means NOT DCIS or LCIS but mucinous features is unrelated to staging]
TUMOR FOCALITY: Multifocal.
Two foci (1.5 and 1.1 mm in greatest dimension)
DISTANCE OF INVASIVE CARCINOMA\nFROM NEAREST MARGIN: 4mm from superior margin.
""",
},
"T1b": {
"Notes": """
Largest Tumor size > 5 mm but <= 10 mm
""",
"Relevant Vocabulary": """
Largest diameter
Major diameter
Largest
Diameter
Size
Greatest
Maximal
Maximum
""",
"Sample Sentences": """
INVASIVE TUMOR SIZE: 5.5 mm
IN-SITU COMPONENT SIZE: 7.0 mm largest contiguous focus
RIGHT BREAST, MASTECTOMY: POORLY DIFFERENTIATED INFILTRATING DUCTAL CARCINOMA, MULTIFOCAL
TUMOR SIZE: 0.7 AND 0.3 CM.
RESECTION MARGIN: NEGATIVE FOR TUMOR (>1.0 CM)
LYMPHOVASCULAR INVASION: NONE SEEN
PERINEURAL INVASION: NONE SEEN
Received fresh and placed in formalin at 16:20. The specimen consists of a breast measuring approximately 21 x 17 x 6m.
There is reddened area on the epidermal surface measuring 0.3 cm. in its greatest dimensions.
This defect measures 2.2 cm. in its greatest dimensions and is located approximately 2 cm. from the deep margin of resection.
DUCTAL CARCINOMA WITH MEDULLARY FEATURES.
TUMOR SIZE: 0.8 IN MAXIMAL DIMENSION.
NOTTINGHAM GRADE: 7 (3 + 3 + 1).
SPECIMEN SIZE: 3 X2.5X1CM.
TUMOR SIZE: GREATEST DIMENSION: 0.8 CM.
TUMOR FOCALITY: SINGLE FOCUS.
MARGINS ARE UNINVOLVED BY INVASIVE CARCINOMA.
CLOSEST MARGIN: MEDIAL MARGIN, 2 MM.
TUMOR SUMMARY: HISTOLOGIC TYPE: Infiltrating ductal carcinoma with apocrine differentiation
TUMOR FOCALITY: Multiple foci of invasive carcinoma (4 foci, measuring 9 mm, 6 mm, 4.5 mm, & 3 mm, respectively)
SIZE OF INVASIVE CARCINOMA: 9 mm is size of largest contiguous focus.
The 6 mm focus of invasion in block H involves the orange inked lateral margin.
This case received intradepartmental consultation with agreemeriE
Received fresh and placed in neutral buffered formalin and consists of one irregular mass of red-tan to yellow-tan fibrofatty tissue measuring 11.0 x 5.3 x 2.9 cm.
The specimen is inked as follows: superior margin - blue; inferior margin - green; medial margin - red; lateral margin - orange; anterior margin - yellow and posterior margin - black.
No additional masses or lesions are grossly identifiable.
A. Superior and inferior with superior being identified with blue ink, inferior with green ink;
B. Lateral and medial margin with lateral being identified with orange ink and medial with red ink;
C. Anterior and posterior with anterior being identified with yellow ink, and posterior with black ink;
D. Represents the nodular area where the wire was located near the medial margin;
E. Represents the nodular area located near the lateral margin;
F, G, and H. Random sections.
MEASURING 1. SURGICAL MARGIN AND 0.8 CM FROM THE ANTERIOR/LATERAL SURGICAL MARGIN
ALSO PRESENT IS DUCTAL CARCINOMA IN-SITU
MEASURING 1.4 CM IN GREATEST DIMENSIONS
SIZE OF INVASIVE CARCINOMA: 0.9 cm
The inked surgical margins are positive for involvement by invasive carcinoma
LYMPHOVASCULAR INVASION: None identified
PERINEURAL INVASION: None identified
TNM STAGING DATA at least pT1b NX
The size of tumor(and thus pathologic staging data) is subject to change since the tumor is transected at the inked margin of specimen
ANCILLARY STUDIES: ER/PR/HER-2/Ki-67 Pending
The specimen consists of two pieces of red-tan to yellow tan fibrofatty tissue which have been sutured together
The specimen measures 2.8 x 2.5 x 1.4 cm
The specimen is inked with blue ink
The specimen is serially sectioned into seven slices revealing a yellow-tan fibrofatty cut surface, has a firm tan-white fibrous appearing area along one edge which measures 0.9 cm in its greatest dimensions
Surgical excision time: Cold Ischemic Time: 2 minutes
This tissue was fixed in neutral buffered formalin between 6 and 48 hours
TUMOR SIZE: 4.2 cm in greatest dimension.
MARGINS: Surgical margins focally positive for invasive carcinoma.
LEFT BREAST, EXCISIONAL BIOPSY:
TUMOR SIZE: 0.6 cm in maximum dimension
TUMOR TYPE: Infiltrating ductal adenocarcinoma.
TUMOR SIZE: 0.9 cm for invasive carcinoma and 1.5 for DCIS
TUMOR GRADE: 1 of 3
TUMOR FOCALITY: Unifocal
LYMPHOVASCULAR INVASION: None seen
PERINEURAL INVASION: None seen
SKIN/ NIPPLE INVOLVEMENT: None seen
SURGICAL MARGIN: Uninvolved by carcinoma with invasive carcinoma and DCIS 0.2 cm from margin
INVASIVE DUCTAL CARCINOMA, MULTIFOCAL AND MULTICENTRIC, SEE TUMOR SUMMARY.
TUMOR SIZE: 0.9 cm. in greatest single dimension.
MARGINS: Margins negative for invasive and in situ carcinoma.
DISTANCE OF INVASIVE AND IN SITU CARCINOMA FROM NEAREST MARGIN: Greater than 0.6 cm.
PATHOLOGIC STAGING: pT1b(m) pNO.
2, SOFT TISSUE, SATELLITE NODULE RIGHT, EXCISION:
CLINICAL INFORMATION:
PREOPERATIVE DIAGNOSIS: Right breast cancer.
The breast tissue with attached pink-tan ellipse of skin measures 19.0 x 14.0 x 5.2 cm.
The nipple is present.
The deep margin is inked with black ink and sectioning through the breast reveals a hemorrhagic cavitary space which resembles a previous biopsy site and measures approximately 3.0 x 2.5 x 2.0 cm and comes to within 0.6 cm of the deep margin of resection.
The specimen is received fresh and placed in a container of neutral buffered formalin and consists of one irregularly shaped piece of yellow-tan to pink fibrofatty tissue measuring 3.0 x 2.0 x 1.5m.
Sectioning through the specimen reveals a hemorrhagic area which measures approximately 1.0 cm in greatest dimension.
MARGINS: MARGINS POSITIVE FOR CARCINOMA
TUMOR FOCALITY: SINGLE FOCUS OF INVASIVE CARCINOMA
TUMOR SUMMARY RIGHT BREAST NEEDLE LOCALIZATION BIOPSY:
HISTOLOGIC TYPE: INVASIVE MAMMARY CARCINOMA OF NO SPECIAL TYPE (DUCTAL CARCINOMA).
TUMOR GRADE, NOTTINGHAM COMBINED HISTOLOGIC SCORE:
TUBULE FORMATION: SCORE 3.
NUCLEAR PLEOMORPHISM: SCORE 3.
MITOTIC COUNT: SCORE 2
OVERALL GRADE: GRADE 3.
MARGINS: MARGINS NEGATIVE FOR CARCINOMA
LEFT BREAST, LUMPECTOMY: INFILTRATING DUCTAL ADENOCARCINOMA.
TUMOR SIZE: 0.7 CM MAXIMAL MEASURED DIMENSION
DUCTAL CARCINOMA IN SITU: PRESENT, EXTENSIVE, HIGH GRADE (NUCLEAR GRADE 3 OF 3)
SURGICAL MARGINS: INVASIVE COMPONENT CLEAR (1.0 CM FROM NEAREST AND INKED INFERIOR MARGIN)
LEFT BREAST LUMPECTOMY WITH NEEDLE WIRE LOCALIZATION: RESIDUAL INFILTRATING DUCTAL CARCINOMA WITH ADJACENT HEALING BIOPSY SITE.
TUMOR SIZE: ESTIMATED AT SLIGHTLY GREATER THAN 0.5 cm IN GREATEST DIMENSION.
EXCISION MARGINS: NEGATIVE FOR TUMOR (APPROXIMATELY 0.7 CM FOR BOTH INVASIVE AND IN SITU DISEASE).
TUMOR FOCALITY: SINGLE FOCUS.
""",
},
# "Largest Tumor size > 10 mm but <= 20 mm",
"T1c": {
"Notes": """
Largest Tumor size > 10 mm but <= 20 mm
""",

"Relevant Vocabulary": """
No more than
No larger than
Larger than but no larger than
Size no more
Size no larger
Size equal to
""",
"Sample Sentences": """
RIGHT BREAST MASS; BIRADS 4
DUCTAL CARCINOMA, GRADE 3, NOTTINGHAM COMBINED HISTOLOGIC SCORE = 9/9
INVASIVE CARCINOMA IS PRESENT AT THE ANTERIOR MARGIN NEAR THE INFERIOR
PROGNOSTIC MARGINS ARE ORDERED ON BLOCK C
LEFT BREAST MASS
DUCTAL CARCINOMA WITH MEDULLARY FEATURES, GRADE 3
THE INVASIVE CARCINOMA MEASURES 1.5 X 1.5 X 1.3
WITH EXTENSION INTO LOBULES IS IDENTIFIED AND INVOLVES APPROXIMATELY 5% OF THE TUMOR
A MARKED LYMPHOCYTIC INFILTRATE ACCOMPANIES THE
THE INVASIVE CARCINOMA HAS MANY OF THE FEATURES OF MEDULLARY CARCINOMA
THE TUMOR CELLS HAVE A SYNCYTIAL PATTERN OF GROWTH
AN INTENSE LYMPGHOPLASMACYTIC INFILTRATE IS PRESENT IN THE INVASIVE TUMOR AND SURROUNDING LOBULES
THE TUMOR HAS A HIGH MITOTIC RATE AND GRADE 2 NUCLEI
ALTHOUGH MOST OF THE TUMOR IS WELL CIRCUMSCRIBED, THERE IS A COMPONENT OF THE TUMOR ON SLIDE F IN WHICH TUMOR INFILTRATING ADIPOSE TISSUE IS FOUND
RIGHT BREAST MASS, EXCISIONAL BIOPSY: INVASIVE DUCT CARCINOMA, HIGH GRADE, 1.8 CM
Margins: All margins uninvolved by invasive carcinoma >1.0 cm
LEFT MODIFIED RADICAL MASTECTOMY: INFILTRATING DUCTAL CARCINOMA WITH FOCAL CARTILAGINOUS STROMAL METAPLASIA.
TUMOR SIZE: Microscopic estimate approximately 1.5 om
RESECTION MARGINS: Negative for tumor (greater than 2.0 cm)
LYMPHOVASCULAR INVASION: None seen.
PERINEURAL INVASION: None seen.
TUMOR FOCALITY: Single tumor focus.
LEFT BREAST, LUMPECTOMY SPECIMEN: POORLY DIFFERENTIATED INFILTRATING DUCTAL ADENOCARCINOMA.
SIZE OF INVASIVE COMPONENT: 1.3 cm.
TUMOR FOCALITY: Unifocal.
LYMPHOVASCULAR INVASION: Present.
SURGICAL MARGIN: Lesion extends to the inked margin of resection.
EXTENT OF MARGIN INVOLVEMENT: 3 mm.
The specimen consists of an irregular mass of yellow-tan to yellow red fibrofatty tissue measuring 10.0 x 6.7 x 3.0 cm.
Located near the end of the wire in slices 6 and 7, is a well-circumscribed white-tan shiny firm mass measuring 1.3 cm in its greatest dimensions.
The mass extends to within 1.0 mm of the inked margin of resection.
LEFT BREAST, LUMPECTOMY: INFILTRATING DUCTAL CARCINOMA I
TUMOR SIZE: 1.2x 1.0 x 0.8 cm.
RESECTION MARGINS: Negative for tumor (approximately 1.0 cm for invasive tumor; approximately 0.3 cm for a small Satellite focus of DCIS).
LYMPHOVASCULAR INVASION: None seen.
PERINEURAL INVASION: None seen.
SKIN INVOLVEMENT: None seen
2. ADDITIONAL INFERIOR MARGIN: BENIGN BREAST TISSUE SHOWING SOME FIBROCYSTIC CHANGES.
RIGHT BREAST LUMPECTOMY: INFILTRATING DUCT AL CARCINOMA WITH METAPLASTIC CHANGES (SQUAMOUS DIFFERENTIATION).
TUMOR SIZE:
PREOPERATIVE DIAGNOSIS: Adenocarcinoma right breast.
The specimen consists of one irregular mass of red-tan to yellow-tan soft tissue measuring 2.0 x 1.3 x 1.5 cm.
The specimen consists of one irregular mass of red-tan to yellow-tan fibrofatty tissue which measures 8.4 x 8.6 x 3.2 cm.
The specimen is serially sectioned and completely submitted in cassettes A and 8.
The specimen is serially sectioned beginning at the lateral margin proceeding towards the medial margin.
Sectioning through the breast tissue located more towards the medial margin reveals a cavitary possible biopsy site which measures 0.9 x 1.0 located adjacent to this biopsy site there is some residual tumor present.
The residual tumor measures 1.3 x 1.3 x 1.4 cm. [The term "residual" here is important. It means tumor that is still alive - as opposed to tumor that has been killed by pre-operative chemotherapy (also called neo-adjuvant)]
This mass is located approximately 0.7 cm from the inferior margin of resection.
The mass comes to within 0.3 cm of the medial margin.
One metal clips was found in the area of the biopsy site. [Notice the gramatical error. Clip and Clips are often used interchangeably]
RIGHT BREAST - LUMPECTOMY SPECIMEN - POORLY DIFFERENTIATED INFILTRATING DUCTAL ADENOCARCINOMA.
TUMOR SIZE 1.3xX1.1x1.2¢em.
TUMOR SIZE: 1.5 cm maximal dimension.
TUMOR TYPE: Infiltrating ductal adenocarcinoma.
RIGHT BREAST, LUMPECTOMY: INFILTRATING DUCTAL ADENOCARCINOMA.
TUMOR SIZE: 1.7 cm maximal dimension
SURGICAL MARGINS: Clear (0.9 cm from nearest and superior inked margin).
RIGHT BREAST, LUMPECTOMY: INFILTRATING DUCTAL CARCINOMA.
TUMOR SIZE Estimated as 1.3 cm in greatest dimension.
RESECTION MARGINS: Negative for tumor (all greater than 0.5 cm)
LYMPHOVASCULAR INVASION: None seen.
PERINEURAL INVASION Present.
SKIN INVOLVEMENT: None.
TUMOR FOCALITY: Single focus.
3. RIGHT BREAST TISSUE, ADDITIONAL INFERIOR MARGIN: BENIGN BREAST TISSUE SHOWING NON-PROLIFERATIVE FIBROCYSTIC CHANGES.
The specimen is received fresh, placed in a container of neutral buffered formalin, and consists of a mass of yellow-tan to pink fibrofatty tissue with an attached ellipse of tan-brown skin.
The specimen is received fresh, placed in a container of neutral buffered formalin, and consists of a mass of yellow-tan to pink fibrofatty tissue measuring 5.2 x 4.3 x 1.6 cm.
Invasive ductal carcinoma with medullary features, grade 3, Nottingham histologic score 8/9
The invasive carcinoma is multifocal with 3 distinct areas identified measuring 2 cm, 0.6 cm and 0.3 cm respectively.
Lymphatic space invasion is identified.
The surgical margins are uninvolved by tumor.
Invasive carcinoma is present 0.2 cm from the superior (nearest) margin.
Right breast mass, excisional biopsy: Invasive medullary carcinoma of breast (1 cm), grade 3.
The invasive carcinoma is less than 0.1 cm from the lateral and posterior margins.
No lymphatic space or perineural invasion is identified.
Tumor size: 1. cm.
Histologic grade: Nottingham histologic score Tubular differentiation ; score 3.
Nuclear pleomorphism: Score 3.
Mitotic rate: Score 2 Number of mitoses per 10 HPF =11.
Ductal carcinoma in situ: Focus of extension of medullary carcinoma into an adjacent lobule.
Margins: Margins uninvolved by invasive carcinoma.
Distance from closest margin: < 0.1 cm (posterior and lateral margins).
Lymphovascular invasion: Not identified.
Perineural invasion: Not identified.
TUMOR SIZE: 1.7m.
TUMOR FOCALITY: Single focus of invasive carcinoma.
Received in formalin labeled left breast is a 28 x 17 x 5 left breast with attached axillary contents.
Surface reveals a 2.0 x 1.5 x 1.4 cm firm tan stellate lesion.
The Lesion is closest to the posterior margin at 0.5 cm away.
Modified radical mastectomy: -1.4 cm infiltrating ductal carcinoma, Grade 3 - Modified bloom and Richardson tubule formation score 3, nuclear score 3, mitotic score 2.
No tumor is identified in the inked surgical margin resection - Tumor comes closest to the posterior margin at 3 mm away.
Multiple foci of vascular invasion are identified.
Sections of skin over tumor are unremarkable.
Sections of nipple show foci of vascular invasion.
Random sections of breast show vascular invasion and focal ductal carcinoma in situ, high nuclear grade, comedotype.
Vascular invasion: Multiple foci of vascular invasion are noted
Extent of tumor - Skin: no skin involvement present - Chest wall: no skeletal muscle present
In situ component: Rare focal DCIS present, comedo type, high nuclear grade, EIC negative
Histologic type: Ductal
Histologic grade: 3 (8 points) Nuclear score: 3 Tubal score: 3 Mitotic score: 2
Margins: Margins uninvolved by invasive carcinoma Distance from closest margin: 3 mm from posterior margin
Left breast, lumpectomy: -1.2 cm infiltrating ductal carcinoma with medullary features, Grade 3
No vascular invasion is identified
No tumor is identified in the inked surgical margins of resection
Tumor comes to within 0.5 mm of blue inked anterior margin of resection
Invasive carcinoma size: 1.2 cm
Histologic type: Ductal with medullary features
Histologic grade: 3 (8 points)
Margin: Margins uninvolved by invasive carcinoma
Distance to closest margin 0.5 millimeters from blue inked anterior margin
Pathologic stage: Tlc NX MX
No in situ component present
Breast prognostic markers have been ordered with results to follow shortly in an addendum report
Tumor size: 1.1 x 1.0 x 0.9 cm
ADENOID CYSTIC CARCINOMA OF THE BREAST
MARGINS OF RESECTION ARE NEGATIVE FOR INVOLVEMENT BY CARCINOMA
TUMOR SIZE: 4.3 cm in greatest dimension
HISTOLOGIC TYPE: Adenoid cystic carcinoma
HISTOLOGIC GRADE: Score = 1
GLANDULAR / TUBULAR DIFFERENTIATION: Score = 2
NUCLEAR PLEOMORPHISM: Score = 2
MITOTIC RATE: Score = 2
OVERALL GRADE: Grade 1
TUMOR FOCALITY: Single focus of invasive carcinoma
MARGINS: Margins uninvolved by carcinoma
DISTANCE OF INVASIVE CARCINOMA TO CLOSEST MARGIN: 0.2 cm from the lateral margin
"""
},
# "Largest Tumor size > 20 mm but <= 50 mm",
"T2": {
"Notes": """
Largest Tumor size > 20 mm but <= 50 mm
""",
"Relevant Vocabulary": """
Size
Diameter
Maximum diameter
Maximum size
Tumor size
Invasive
Infiltrating
[NOTE: the masses/lesions whose measurements are relevant are always the invasive components, which are also called infiltrating. Invasive and infiltrating are synonymous]
[NOTE: tumor = cancer = CA = Ca]
[NOTE: "margins" refers to the surgical margins, not the size of the lesion. It means how far away the lesion is from the margins of the surgical excision. If margins are UNINVOLVED it means the tumor was taken out entirely. If margins are INVOLVED, INFILTRATED it means a new surgery is needed. This doesn't enter into staging]
""",
"Sample Sentences": """
RIGHT BREAST MASS
RIGHT POSTERIOR MARGIN
INVASIVE CARCINOMA MEASURES 2.5 X 2.3 X 2.2 CM.
INVASIVE CARCINOMA IS FOCALLY <1 MM FROM THE POSTERIOR (NEAREST) MARGIN.
SMALL AREAS OF DUCTAL CARCINOMA IN-SITU HIGH NUCLEAR GRADE, WITH EXTENSION INTO LOBULES, ARE NOTED ADJACENT TO THE INVASIVE CARCINOMA.
BENIGN FATTY TISSUE.
RIGHT BREAST CA
METASTATIC POORLY DIFFERENTIATED INFILTRATING DUCTAL CARCINOMA
MEASURING 5 CM IN GREATEST GROSS DIMENSIONS
TOTAL NOTTINGHAM SCORE OF 9 (TUBULE FORMATION = 3, MITOTIC COUNT = 3, NUCLEAR PLEOMORPHISM = 3)
100% OF THE TUMOR CELLS STAIN WITH 4+ POSITIVITY FOR E-CADHERIN
2.1% POSITIVITY FOR ESTROGEN RECEPTOR
0.37%% FOR PROGESTERONE RECEPTOR
0+ POSITIVITY FOR HER 2/NEU
RIGHT BREAST, LUMPECTOMY: INVASIVE DUCT CARCINOMA, HIGH GRADE.
TUMOR SIZE: 2.3 x 2.2 x 2.1 CM
4, ADDITIONAL POSTERIOR MARGIN: MARGIN IS FREE.
PREOPERATIVE DIAGNOSIS: Right breast mass.
3. The specimen was received unfixed and consists of a portion of breast tissue measuring 8.5 x 6.2 x 2.8 cm. The specimen is oriented and the margins are inked as follows: anterior - blue; posterior - black; lateral - yellow; medial - orange; superior - red; inferior - green. After inking the margins, the specimen was sectioned and the gross margins were noted as follows: anterior margin - 1.5 cm; posterior margin - 0.1 cm; lateral margin - 2 cm; medial margin - 2.5 cm; superior margin - 1.3 cm; inferior margin, green - 1.1 cm. One fixed section is submitted for breast protocol. Additional sections will be submitted following fixation.
3. Post-fixation, the specimen is sectioned and reveals an irregular gray-tan nodule, 2.3 x 2.2 x 2.1 cm in maximal dimensions. Representative portions to include the margins of the lesion are submitted in seven cassettes.
4. The specimen consists of a re-excision of breast margin 9 x 4.2 x 1.8 cm. There is no grossly visible tumor. The inner surface is stained with blue dye. Representative portions are submitted in two cassettes.
RIGHT BREAST, BIOPSY
+ MEDULLARY CARCINOMA
Tumor focally is identified extending to an inked margin of excision
The specimen consists of a yellow-tan fibrofatty soft tissue measuring 4.5 x 3.2 x 1.5 cm
Sectioning reveals an ill-defined yellow-tan to red-tan shiny area which is more evident in slices 5, 6, 7 and 8
This area measures 2.8 cm in its greatest dimensions and comes within 2 mm of the inked margin
LEFT BREAST NODULE, WIRE LOCALIZATION EXCISIONAL BIOPSY: 2- INVASIVE DUCTAL CARCINOMA WITH MEDULLARY FEATURES, GRADE 3, NOTTINGHAM COMBINED HISTOLOGIC SCORE 8/9
ALL OF THE MARGINS ARE FREE
LEFT BREAST CARCINOMA WITH MEDULLARY FEATURES, GRADE 3, NOTTINGHAM COMBINED HISTOLOGIC SCORE 8/9
RIGHT BREAST, RADICAL MASTECTOMY: INVASIVE DUCTAL CARCINOMA, GRADE 2.
Tumor size: 4.5 x 2.5 x 2.0m
Margins: All margins uninvolved by invasive carcinoma.
LEFT BREAST, EXCISIONAL BIOPSY:
+ MEDULLARY CARCINOMA.
TUMOR SIZE: 3.6 x 3.0 x 1.9m.
LEFT BREAST INVASIVE DUCTAL CA
THE INVASIVE CARCINOMA ARISES FROM HIGH-GRADE DUCTAL CARCINOMA IN-SITU
AND IS 1.5 CM DEEP TO THE SKIN SURFACE
AJCC 7TH EDITION PATHOLOGIC CLASSIFICATION PT2
NO DERMAL LYMPHATIC SPACE INVASION OR PAGETOID SPREAD IN THE SKIN IS IDENTIFIED
NO LYMPHOVASCULAR INVASION OR PERINEURAL INVASION IS IDENTIFIED
CARCINOMA WITH FOCAL SQUAMOUS AND SPINDLE CELL FEATURES
GRADE 3, NOTTINGHAM COMBINED HISTOLOGIC SCORE = 9/9
TUMOR SIZE <= 20 mm
NO TUMORAL CALCIFICATION IS SEEN
LEFT BREAST MASS
EXCISIONAL BIOPSY OF THE BREAST
IRREGULAR TO HARD NODULE
INVASIVE WITH CARCINOMA
MARGINS ARE FREE
UNINVOLVED BY INVASIVE CARCINOMA
LEFT BREAST MASS
EXCISIONAL BIOPSY OF TAN-PINK SOFT TISSUE MEASURING 6 X 2.9 X 2.4 CM
A WHITE-PINK CYSTIC AREA IS IDENTIFIED, MEASURING 0.3 CM IN GREATEST DIMENSIONS
A SECOND OVOID, FIRM MASS IS IDENTIFIED MEASURING 2 CM IN GREATEST DIMENSIONS
THE TUMOR IS 0.4 CM FROM THE ANTERIOR SURGICAL MARGIN
DIFFERENTIATED INFILTRATING DUCTAL CARCINOMA, TOTAL NOTTINGHAM SCORE OF 8
MEASURING 2 CM IN GREATEST
FROM THE INFERIOR, DEEP, AND LATERAL SURGICAL MARGINS
0.5 CM FROM THE SUPERIOR AND ANTERIOR SURGICAL MARGINS
GREATER THAN 0.5 CM FROM THE MEDIAL SURGICAL MARGIN
Tumor Size: 3.9m
Margins: Margins uninvolved by invasive carcinoma.
TUMOR SIZE 2.7 cm in greatest dimension
RESECTION MARGINS: Negative for tumor (nearest margin inferior is 0.2 cm).
LYMPHOVASCULAR INVASION: None seen.
PERINEURAL INVASION: None seen.
STAGING CODE: PT2 NO G3 of 3.
PREOPERATIVE DIAGNOSIS: Right breast mass, right breast cancer.
This residual tumor appears to come within 0.2 cm of the nearest and inferior margin of resection.
The specimen consists of one irregular mass of red-tan to yellow-tan fibrofatty tissue which measures 4.0 x 3.4 x 1.1 cm.
No masses or lesions are grossly identifiable
The specimen consists of one irregular mass of red-tan to pink soft tissue which measures 1.0 x 0.8 x 0.9 cm.
RIGHT BREAST MASS
EXCISIONAL BIOPSY WITH WIRE-GUIDED LOCALIZATION
INVASIVE DUCTAL CARCINOMA, 23 MM IN GREATEST DIMENSION
NO DUCTAL CARCINOMA IN-SITU IS IDENTIFIED
UNINVOLVED BY INVASIVE CARCINOMA
CLOSEST MARGIN: INFERIOR
DISTANCE FROM CLOSEST MARGIN: 1.2 MM
NO LYMPHOVASCULAR INVASION IS IDENTIFIED
NO PERINEURAL INVASION IS IDENTIFIED
TUMOR SIZE OF LARGEST INVASIVE CARCINOMA: 34 mm.
ANCILLARY STUDIES: ER - negative, PR - negative, HER2 - negative, Ki-67 - unfavorable per previously reported results (AT12-23148),
The smaller tumors were located at least 45 mm away from the largest mass.
The smallest (grossly inapparent) tumor approaches to within 0.2 cm of inferior margin with the mid sized tumor approaches no closer than 17 mm from the nearest margin (inferior margin), and the largest tumor approached no closer than 10 mm from the nearest margin (anterior).
The larger tumor is used for staging purposes in accordance with AJCC guidelines.
The smaller tumors showed similar histologic features including similar grade, absence of lymphovascular invasion, and overall growth pattern as the largest tumor.
INFILTRATING DUCTAL ADENOCARCINOMA OF THE BREAST, GRADE 3 OF 3
MARGINS FREE OF TUMOR
SIZE OF INVASIVE COMPONENT
SIZE, TYPE, EXTENT OF INTRADUCTAL COMPONENT
SURGICAL MARGINS
DISTANCE TO CLOSEST MARGIN
EXTENT OF MARGIN INVOLVEMENT
Following Quality Assurance (QA) review, a second microscopic focus (0.3 cm) of invasive and in situ carcinoma is identified, which was not noted in the initial report.
This finding changes the final corrected staging to pT2(m) pNX.
BREAST, LEFT BREAST MASS, NEEDLE LOCALIZATION WIRE GUIDED BIOPSY: INVASIVE DUCTAL CARCINOMA, NOTTINGHAM COMBINED HISTOLOGIC GRADE 3 (3,3,2), 2.4 CM.
DUCTAL CARCINOMA IN SITU, HIGH NUCLEAR GRADE.
SURGICAL MARGINS ARE INVOLVED BY INVASIVE CARCINOMA.
TUMOR SUMMARY LEFT BREAST NEEDLE LOCALIZATION BIOPSY:
HISTOLOGIC TYPE: INVASIVE DUCTAL CARCINOMA.
TUMOR SIZE: 2.4CM.
TUMOR GRADE/ NOTTINGHAM HISTOLOGIC SCORE:
GLANDULAR DIFFERENTIATION: SCORE 3.
NUCLEAR PLEOMORPHISM SCORE 3.
MITOSES: SCORE 2.
OVERALL GRADE: GRADE 3.
TUMOR FOCALITY: TWO FOCI OF INVASIVE CARCINOMA.
DISTANCE OF INVASIVE CARCINOMA FROM CLOSEST MARGIN: SEE COMMENT.
ANCILLARY STUDIES: ESTROGEN RECEPTOR: NEGATIVE
PROGESTERONE RECEPTOR: NEGATIVE
HER 2: NEGATIVE BY IMMUNOHISTOCHEMISTRY AND IN SITU HYBRIDIZATION
Comment: While the needle localization biopsy (specimen #1) shows invasive carcinoma at the margin of excision, there is no residual tumor in the re-excision specimen (specimen #2).
As both specimens were unoriented, the measurement of carcinoma to the final true surgical margin is not possible, but is likely to be at least 3.5 cm (the smallest dimension of the cavity specimen).
LEFT BREAST - UPPER QUADRANT - LUMPECTOMY SPECIMEN - POORLY DIFFERENTIATED INFILTRATING DUCTAL ADENOCARCINOMA.
SIZE OF INVASIVE COMPONENT 3.5 x 2.6 x 2.2 cm.
LYMPHOVASCULAR INVASION Present and extensive.
DISTANCE TO CLOSEST MARGIN Less than 1 mm. - posterior margin.
LEFT BREAST LESION 3 O'CLOCK, EXCISED AT 1305, IN FORMALIN AT 1315
LEFT UPPER QUADRANT BREAST MASS, EXCISED AT 1350, IN FORMALIN AT 1400
LEFT BREAST - 3 O'CLOCK LESION - SKIN AND BREAST TISSUE - PROLIFERATIVE BREAST DISEASE WITH ADENOSIS AND FIBROSIS.
TUMOR FOCALITY Unifocal.
SPECIMEN TYPE Lumpectomy specimen
HISTOLOGICAL TYPE Poorly differentiated infiltrating ductal adenocarcinoma,
NOTTINGHAM HISTOLOGICAL SCORE 3+343.
DISTANCE TO CLOSEST MARGIN Less than 1 mm. - posterior margin.
LEFT BREAST AND AXILLARY CONTENTS - POORLY DIFFERENTIATED INFILTRATING DUCTAL ADENOCARCINOMA.
SIZE OF INVASIVE COMPONENT: 4.0x3.5x3.2cm.
TUMOR FOCALITY: Unifocal.
TUMOR NECROSIS: Present - extensive.
VASCULAR INVASION: None seen.
PERINEURAL INVASION: None seen.
CALCIFICATION Present.
SKIN/NIPPLE INVOLVEMENT: None seen
SURGICAL MARGINS: Free of disease.
DISTANCE TO CLOSEST SURGICAL MARGIN: 1.0m,
BREAST PROGNOSTIC PANEL: See AS13-22060 (ER negative/PR negative/Her2-neu negative).
LEFT BREAST AND AXILLARY CONTENTS
Received in neutral buffered formalin and consists of a mastectomy specimen with axillary tail 1,329 grams and measuring 20.5 x 18.3 x 7.5 cm.
The skin surface is tan-brown with a previous incision site within the inferior lateral aspect measuring 3.2 cm.
The nipple is elevated measuring 1.3 cm in diameter.
Sectioning reveals a tan-white mass measuring approximately 4.0 x 3.5 x 3.2 cm, appearing to extend to and / or involve the black inked deep margin.
Adjacent to the mass is an area of previous site correlating with the described likely incision on skin surface measuring 5.0 x 4.0 x 4.0 cm.
The mass and biopsy site are located within the inferior lateral aspect of the specimen.
INFILTRATING DUCTAL CARCINOMA, NOTTINGHAM GRADE 3, AND INTERMEDIATE GRADE DCIS WITH COMEDONECROSIS.
INVASIVE TUMOR SIZE: 26 mm.
TUMOR FOCALITY: Single focus of invasive carcinoma present.
SKIN: Skin present and uninvolved by invasive carcinoma.
SURGICAL MARGINS: Positive for involvement by invasive carcinoma (focally along anterior margin).
RIGHT BREAST CANCER
THE SPECIMEN CONSISTS OF A LYMPH NODE AND REPRESENTATIVE SOFT TISSUE 2.3 X 1.6 X 1 CM IN MAXIMAL DIMENSIONS.
THE NODE FEELS FIRM. CUT SURFACE IS GRAYISH-WHITE AND HOMOGENOUS.
THE SPECIMEN CONSISTS OF A LYMPH NODE AND REPRESENTATIVE SOFT TISSUE 1.5 X 0.9 X 0.8 CM.
THE NODE FEELS FIRM AND GRAYISH-WHITE. CUT SURFACE IS GRAYISH-WHITE AND HOMOGENOUS.
THE ENTIRE SPECIMEN IS 29 X 21X 5.5 CM.
THE BREAST IS 20 X 16 X 4CM.
THE NIPPLE IS 1 CM IN MAXIMAL DIMENSION AND ALMOST COMPLETELY RETRACTED.
SITUATED 1 CM FROM THE DEEP MARGIN, 3 CM FROM THE INFERIOR MARGIN, 8 CM FROM THE SUPERIOR MARGIN, 6 CM FROM THE LATERAL MARGIN, 1.5 CM FROM THE ANTERIOR MARGIN AND 9 CM FROM THE MEDIAL MARGIN IS A SPICULATED FIRM TO HARD POORLY DEFINED NODULE 4.4 X 4.2 X 4 CM IN MAXIMAL DIMENSION.
INVASIVE DUCTAL CARCINOMA, APOCRINE TYPE, GRADE 3
SKELETAL MUSCLE: CARCINOMA INVADES SKELETAL MUSCLE
MARGINS UNINVOLVED BY INVASIVE CARCINOMA
TUMOR TYPE: Infiltrating duct carcinoma with focal squamous differentiation
HISTOLOGIC GRADE: High grade.
SIZE OF INVASIVE COMPONENT: 3.5 cm maximally.
TUMOR FOCALITY: Unifocal.
TUMOR NECROSIS: Present but not extensive.
LYMPHOVASCULAR INVASION: None seen
PERINEURAL INVASION: None seen
Left breast mass, excision with wire-guided localization: Invasive ductal carcinoma, grade 3, Nottingham histologic score 8/9 (tubule formation= 3, nuclear pleomorphism= 2, mitotic frequency= 71/10 HPF = 3).
The invasive carcinoma measures 3.8 cm in greatest dimension.
Ductal carcinoma in situ, solid type, nuclear grade 3, is present adjacent to and separate from the invasive carcinoma.
Invasive carcinoma is present at the superior margin.
LEFT BREAST, LUMPECTOMY: - INFILTRATING POORLY DIFFERENTIATED DUCT CARCINOMA.
BREAST, RIGHT, LUMPECTOMY:
TUMOR LESS THAN 0.1 CM FROM DEEP MARGIN
Left breast, simple mastectomy: Benign breast tissue with multifocal duct ectasia and stromal fibrosis. Focal cystic apocrine metaplasia No atypia or malignancy is identified in the specimen.
Right breast, simple mastectomy: Invasive ductal carcinoma, grade 3, Nottingham histologic score 9/9 (tubule formation = 3, nuclear pleomorphism = 3, mitotic activity = 3). The invasive carcinoma measures 2.8 cm in greatest dimension and has extensive central necrosis. Focal ductal carcinoma in situ (DCIS), high nuclear grade, is present adjacent to the invasive carcinoma and represents < 1% of the tumor. No in situ or invasive carcinoma is identified at any of the surgical margins. Invasive carcinoma and DCIS are at least 1 cm from the nearest (superior) margin. No lymphatic space or perineural invasion is identified.
Invasive ductal carcinoma
Extensive high-grade ductal carcinoma in situ with comedonecrosis
The black inked superior margin is positive for high-grade ductal carcinoma in situ
Invasive ductal carcinoma
Grade (modified Bloom Richardson score): 3
Tubule score: 3
Nuclear score: 3
Mitotic score: 3
Lymphovascular invasion is identified on sections examined
Margins are negative for invasive ductal carcinoma on sections examined
Superior, 2mm
Inferior, 1 mm
Invasive carcinoma size: 2.2 cm (specimen #3) and 2 cm (specimen #1)
Histologic grade: 3
Margins uninvolved by invasive carcinoma
Superior, 2mm
Inferior, 1 mm
Margins involved by ductal carcinoma in situ
Lymphovascular invasion is identified on specimen #3
ER: Negative, with 0% tumor cells showing nuclear staining
PR: Negative, with 0% tumor cells showing nuclear staining
HER-2/neu: negative, with 0 tumor cell membrane staining
ER: Negative, with 0% tumor cells showing nuclear staining
PR: Negative, with 0% tumor cells showing nuclear staining
HER-2/neu: negative, with 0 tumor cell membrane staining
TUMOR SIZE: 3.5 X 2.7 X 2.5 CM
TREATMENT EFFECTS: NONE.
PECTORALIS MUSCLE NEGATIVE FOR TUMOR.
Right breast mass
Invasive ductal carcinoma, grade 3
The invasive carcinoma measures 2.2 cm in greatest dimension
Lymphatic space invasion is present
Ductal carcinoma in situ (DCIS), high nuclear grade, solid and comedo type
Multifocal DCIS involvement of lobules is present
No invasive carcinoma or DCIS involves the surgical margins
Histologic grade: Grade 3
Tumor focality: Single focus of invasive carcinoma
Margins uninvolved by invasive carcinoma
Lymphovascular invasion: Present
TUMOR TYPE: Invasive ductal carcinoma, not otherwise specified.
INVASIVE MAMMARY CARCINOMA - SEE TUMOR SUMMARY.
TUMOR SUMMARY, RIGHT BREAST LUMPECTOMY AND SENTINEL NODE BIOPSY:
MARGINS: All margins negative for invasive carcinoma.
Right breast, lumpectomy: -Invasive ductal carcinoma, Grade 3
Largest area of invasive tumor measures 4.6 x 4.2 x 3.5 cm
TUMOR SUMMARY (CONSIDERING PARTS 1 AND 2):
TUMOR TYPE: Infiltrating duct carcinoma.
TUMOR FOCALITY: Unifocal.
Invasive ductal carcinoma, Grade 2
Largest area of invasive tumor size measures 2.8 x 1.2 x 1.0 cm
Second area of invasive tumor size measures 0.5 x 0.4 x 0.3 cm
Received fresh and subsequently placed in formalin labeled with the patient's name, medical record number and designated right breast is a 972 g mastectomy specimen designated with 2 sutures as follows: Long lateral, short superior.
The specimen measures 24.5 x 21 x 3.8 cm.
Serial sectioning reveals a firm white tan lesion located at 8:00 position measuring 2.2 x 2.0 x 2.0 cm.
The lesion is 2 mm from posterior margin and widely clear of all other margins.
Breast (972g), right, mastectomy: -Invasive ductal carcinoma -Grade (modified Bloom Richardson score): 3 of 3 -Nuclear score: 3 of 3 -Mitotic score: 3 of 3 -Tubular score: 3 of 3 -Margins: -Margins are free of carcinoma on sections examined -Closest margin: Invasive carcinoma measures 2 mm from its closest posterior margin __-Note: Skeletal muscle present at the posterior aspect of the specimen
Pathologic staging (pTNM): pT2 NO Mx pT2: Tumor greater than 20 mm but less than or equal to 50 mm in greatest dimension pNO: No regional lymph node metastasis identified histologically pMx: Not applicable
Prognostic markers performed on block 2B Results are as follows: HORMONE RECEPTOR STUDIES: THE CARCINOMA IS NEGATIVE FOR NUCLEAR ESTROGEN RECEPTORS, NUCLEAR PROGESTERONE RECEPTORS, AND. HER-2.
Serial sectioning through the posterior aspect of the breast reveals a well-circumscribed firm yellow-tan mass measuring 2.8 x 3.1 x 3.3 cm located in the upper outer quadrant with the following margins: Superior 3 cm, inferior 5.2 cm, posterior 1 mm, anterior 8 mm, lateral 5 cm, and medial 8 cm.
Breast, left, mastectomy: -Invasive ductal carcinoma, not otherwise specified, 3.3 x 3.1x 2.8 cm, located in the upper outer quadrant.
The surgical margins are uninvolved by invasive carcinoma; distance from closest margin: 2 mm (posterior margin).
Pathologic staging: pT2pNO(i+)
TUMOR SIZE: 2.3 CM.
TUMOR SUMMARY LEFT BREAST LUMPECTOMY WITH AXILLARY LYMPH NODE BIOPSY:
TUMOR SIZE: Greater than 2.0 cm but less than 4.0 cm.
TUMOR FOCALITY: Unifocal.
LYMPHOVASCULAR INVASION None seen.
PERINEURAL INVASION: None seen.
INTRADUCTAL COMPONENT: None seen.
TUMOR GRADE: HIGH GRADE (3 OF 3).
RESECTION MARGINS: NEGATIVE FOR TUMOR (DEEP MARGIN 1.0 CM FOR INVASIVE CA. AND DCIS).
SKIN AND NIPPLE INVOLVEMENT: NONE.
""",
},
# "Largest Tumor size > 50 mm",
"T3": {
"Notes": """
Largest Tumor size > 50 mm
""",
"Relevant Vocabulary": """
Same as above - except the numbers indicating tumor size will be >5cm
""",
"Sample Sentences": """
RIGHT MODIFIED RADICAL MASTECTOMY:
POORLY DIFFERENTIATED INFILTRATING DUCTAL CARCINOMA.
TUMOR SIZE: 9.2 x 7.6 x 3.9. cm.
RESECTION MARGINS: Negative for tumor (deep margin 0.2 cm).
SKIN INVOLVEMENT: Focal dermal lymphatic invasion present, but no ulceration or Paget's disease.
DCIS: Present, high grade, extensive.
LYMPHOVASCULAR INVASION: Present, extensive.
PERINEURAL INVASION: None seen.
TUMOR FOCALITY: Second small satellite nodule present (0.8 cm).
This histologic finding does not qualify for a T4 stage.
If there is clinical evidence of inflammatory carcinoma, the stage could be changed to T4d
TUMOR SIZE: 13.2 x 11.6 x 10.3 CM; 1.7 CM SATELLITE TUMOR.
TUMOR FOCALITY: TWO FOCI
TUMOR GRADE: HIGH GRADE (3 OF 3), BOTH LESIONS.
SKIN/NIPPLE INVASION: NONE
SKELETAL MUSCLE INVASION: PRESENT IN RELATION TO EXTRANODAL EXTENSION (SEE COMMENT).
RESECTION MARGINS: NEGATIVE FOR TUMOR; APPROXIMATELY 0.2 CM FOR INVASIVE CARCINOMA IN\nRELATION TO THE SUPERIOR AND DEEP SOFT TISSUE MARGINS
LYMPHOVASCULAR INVASION: — PRESENT.
The identified area of invasion of the pectoralis muscle was\nrelated to extranodal extension from a lymph node metastasis.
Multiple sections of the primary tumor do not show direct extension into the\npectoralis muscle, For this reason, the tumor was staged as a T3.

""",
},
# "Largest Tumor with chest wall invasion",
"T4a": {
"Notes": """
Largest Tumor with chest wall invasion
[These are tumors that invade the muscles or skin of the chest wall]
""",
"Relevant Vocabulary": """
Muscle 
Pectoralis muscle
Serratus muscle
Chest wall
Skin invasion
""",

"Sample Sentences": """
Chest wall involved
Chest wall involvement
Chest wall invasion
Involvement of pectoralis muscle noted
Involvement of serratus muscle noted
Positive for chest wall invasion
Positive for serratus muscle invasion
Positive for skin invasion
Skin involvement noted
Skin is involved
Serratus muscle is involved
Pectoralis muscle is involved
""",
"Negative Sample Sentences": """
Negative for chest wall invasion
Negative for chest wall involvement
Chest wall uninvolved
Chest wall invasion (-)
No chest wall invasion
No muscle invasion
No muscle involvement is noted
No chest wall invasion is noted

""",
},
# "Largest Tumor with macroscopic skin changes including ulceration and/or satellite skin nodules and/or edema",
"T4b": {
"Notes": """
Largest Tumor with macroscopic skin changes including ulceration and/or satellite skin nodules and/or edema
[These are tumors that cause visible skin ulceration or visible nodules in the skin]
""",

"Relevant Vocabulary": """
Skin ulceration noted
Skin is ulcered
The tumor invades the skin with ulceration noted
Cutaneous ulcer noted
Cutaneous invasion noted with ulceration
Cutaneous noduls are noted
Positive for cutaneous nodules
Cutaneous tumor nodules are present
Cutaneous tumor nodules are noted
Cutaneous nodules and ulceration are noted
""",
"Sample Sentences": """
TUMOR SIZE: 11.5 CM IN GREATEST DIMENSION (SECOND AXILLARY TAIL NODULE = 9.5 CM)
SKIN: INVASIVE CARCINOMA DIRECTLY INVADES SKIN WITH ULCERATION
MARGINS: MARGINS NEGATIVE FOR INVASIVE CARCINOMA
DISTANCE OF INVASIVE CARCINOMA FROM NEAREST MARGIN: 2.0 CM FROM DEEP MARGIN

""",
},
# "Largest Tumor meets criteria of both T4a and T4b",
"T4c": {
"Notes": """
Largest Tumor meets criteria of both T4a and T4b
[These are tumors that BOTH invade the skin/chest wall and muscles AND cause ulceration and/or nodules in the skin]
""",
"Relevant Vocabulary": """
Muscle involvement
Pectoralis muscle involvement
Chest wall invasion
Chest wall involvement
AND -
All vocabulary from above
""",
"Sample Sentences": """
Chest wall invasion with cutaneous ulceration
Chest wall invasion with cutaneous nodules
Chest wall invasion with skin ulceration
Chest wall invasion with skin nodules
Pectoralis muscle involvement with cutaneous ulceration
Serratus muscle involvement with cutaneous nodules
Tumor involves the chest wall with cutaneous ulcers
Tumor involves the pectoralis muscle and the skin with ulceration and nodules
Tumor involves the serratus muscle and the skin with visible cutaneous ulceration and satellite nodules
""",
"Negative Sample Sentences": """
Tumor involves the chest wall but does not present with cutaneous ulceration(s) or cutaneous nodules
Chest wall involvement without skin ulceration
Chest wall invasion but negative for cutaneous ulceration
Chest wall is involved but no cutaneous ulceration is noted
""",
},
# "Inflammatory carcinoma" # No references available for this case
"T4d": {
"Notes": """
[These are tumors that infiltrate the lymphatic vessels of the skin, causing reddening that looks like inflammation]
""",
"Relevant Vocabulary": """
Any use of the word inflammatory or inflammatory carcinoma
Dermal lymphatics
Involvement of dermal lymphatics
""",

"Sample Sentences": """
Involvement of dermal lymphatics with inflammatory features
Tumor shows features of inflammatory carcinoma
Invasion of dermal lymphatics present with inflammatory features
Tumor invades dermal lymphatics
""",

"Negative Sample Sentences": """
No inflammatory carcinoma present
No inflammatory features
Inflammatory features are absent
Absence of inflammatory features
No invasion of dermal lymphatics present
No inflammatory carcinoma present
Inflammatory features absent
"""
}
},
m_lookup = {
# "Metastases cannot be assessed (Check with Dr. Miele, as this is inferred and not in provided tables)",
"Mx": {
"Notes": """
Metastases cannot be assessed
[These are generally tumors where the main tumor has been excised but there are no other tests in the chart indicating whether or not metastases are present in other organs, such as the bone, the brain, the liver, the lungs etc.]
""",
"Relevant Vocabulary": """
Metastases not assessed
No assessment of metastasis 
Metastatic status unassessed
""",

"Sample Sentences": """
LEFT BREAST - 3 O'CLOCK LESION - SKIN AND BREAST TISSUE - PROLIFERATIVE BREAST DISEASE WITH ADENOSIS AND FIBROSIS.
LEFT BREAST LESION 3 O'CLOCK, EXCISED AT 1305, IN FORMALIN AT 1315
LEFT UPPER QUADRANT BREAST MASS, EXCISED AT 1350, IN FORMALIN AT 1400

""",
},
# "No clinical or imaging evidence of distant metastases",
"M0": {
"Notes": """
No clinical or imaging evidence of distant metastases
[These are cases where metastases WERE looked for and not found]
""",
"Relevant Vocabulary": """
Absent
Absence
No evidence of metastasis
Undetectable
""",
"Sample Sentences": """
NO CLINICAL OR IMAGING EVIDENCE OF DISTANT METASTASES
DISTANT METASTASES ON THE BASIS OF CLINICAL OR IMAGING FINDINGS
HISTOLOGICALLY PROVEN DISTANT METASTASES IN SOLID ORGANS
NO CLINICAL OR IMAGING EVIDENCE OF DISTANT METASTASES
NO CLINICAL OR IMAGING EVIDENCE OF DISTANT METASTASES, BUT WITH TUMOR CELLS OR DEPOSITS MEASURING <= 0.2 mm DETECTED IN CIRCULATING BLOOD, BONE MARROW, OR OTHER NONREGIONAL NODAL TISSUE IN THE ABSENCE OF CLINICAL SIGNS AND SYMPTOMS OF METASTASES
DISTANT METASTASES ON THE BASIS OF CLINICAL OR IMAGING FINDINGS
HISTOLOGICALLY PROVEN DISTANT METASTASES IN SOLID ORGANS; OR, IF IN NONREGIONAL NODES, METASTASES MEASURING >0.2 mm
NO CLINICAL OR IMAGING EVIDENCE OF DISTANT METASTASES
NEGATIVE FOR METASTASIS
DISTANT METASTASIS: NOT APPLICABLE
NO CLINICAL OR IMAGING EVIDENCE OF DISTANT METASTASES
No clinical or imaging evidence of distant metastases
No clinical or imaging evidence of distant metastases
Histologically proven distant metastases in solid organs
No clinical or imaging evidence of distant metastases
Distant metastases on the basis of clinical or imaging findings
Histologically proven distant metastases in solid organs
""",
},
# "No clinical or imaging evidence of distant metastases, but with tumor cells or deposits measuring <= 0.2 mm detected in circulating blood, bone marrow, or other nonregional nodal tissue in the absence of clinical signs and symptoms of metastases",
# Special Case with parent category cM0
"cM0(i+)": {
"Notes": """
No clinical or imaging evidence of distant metastases, but with tumor cells or deposits measuring <= 0.2 mm detected in circulating blood, bone marrow, or other nonregional nodal tissue in the absence of clinical signs and symptoms of metastases
Meets the same conditions of M0, 
These are tumors where small clumps of tumor cells are detected in the blood, the bone marrow or distant lymph nodes
""",

"Relevant Vocabulary": """
Tumor cells
Tumor cells detected
Nonregional nodes
Bone marrow
Blood
Circulating
""",
"Sample Sentences": """
No clinical evidence of metastasis but tumor cells are present in nonregional nodes
No clinical evidence of metastasis but tumor cells noted in bone marrow biopsy
No clinical evidence of metastasis but circulating tumor cells present
Circulating tumor cells are noted with no clinical evidence of metastasis
No evidence of metastasis, circulating tumor cells were detected

""", 
},
# "Distant metastases on the basis of clinical or imaging findings",
"cM1": {
"Notes": """
Distant metastases on the basis of clinical or imaging findings
These are tumors whereby metastases were sought AND found. By definition, they are all stage 4
""",
"Relevant Vocabulary": """
Distant metastasis
Bone metastasis
Metastasis
Brain metastasis
Visceral metastasis
Metastatic
""",
"Sample Sentences": """
Positive for metatastsis
Metastatic involvement of...chest
Metastatic spread to chest
Presence of distant metastases in chest
Imaging shows metastatic nodules in chest
Radiographic evidence of metastatic lesions in chest
Metastatic involvement of chest
Distant metastases detected
Distant metastases demonstrated by [name technique here]
""",
},
# "Histologically proven distant metastases in solid organs; or, if in nonregional nodes, metastases measuring >0.2 mm"
"pM1": {
"Notes": """
Histologically proven distant metastases in solid organs; or, if in nonregional nodes, metastases measuring >0.2 mm
These are tumors whereby metastases were seen radiologically AND biopsied AND proven to be metastases by pathology. Hence the prefix p as opposed to c
""",
"Relevant Vocabulary": """
Confirmed
Demonstrated
Proven
Pathologically
Biopsy
""",
"Sample Sentences": """
Pathologically confirmed metastatic involvement of chest
Pathologically proven metastatic lesions in chest
Biopsy confirmed metastatic lesions
Metastatic nodule, pathologically confirmed
Metastatic involvement, pathologically confirmed

"""
}
}
n_lookup = {
# "Regional nodes cannot be assessed (previously removed)",
"cNx": {
"Notes": """
Regional nodes cannot be assessed (previously removed)
These are tumors where CLINICAL assessment of lymph nodes was not possible because they had been removed surgically
""",
"Relevant Vocabulary": """
Cannot be assessed
Unassessed
Not assessable
""",
"Sample Sentences": """
LEFT BREAST - 3 O'CLOCK LESION - SKIN AND BREAST TISSUE - PROLIFERATIVE BREAST DISEASE WITH ADENOSIS AND FIBROSIS.
LEFT BREAST LESION 3 O'CLOCK, EXCISED AT 1305, IN FORMALIN AT 1315
LEFT UPPER QUADRANT BREAST MASS, EXCISED AT 1350, IN FORMALIN AT 1400
LYMPH NODES EXAMINED: 0
ADDITIONAL PATHOLOGIC FINDINGS: Proliferative fibrocystic disease to include usual ductal hyperplasia, apocrine metaplasia, and fibrosis.

""",
},
# "No regional nodal metastases",
"cN0": {
"Notes": """
No regional nodal metastases
These are tumors where CLINICALLY no swelling is felt indicating likely absence of metastasis
""",
"Relevant Vocabulary": """
No 
Absent
Absence
Negative
Undetectable
""",
"Sample Sentences": """
LYMPH NODES: No metastatic lesions noted
Absence of lymph node metastasis
Metastatic spread to regional lymph nodes absent
No regional lymph node metastasis clinically detectable
Regional node metastasis clinically undetectable
""", 

"Negative Sample Sentences": """
Clinical evidence of regional node metastasis
Possible regional node metastasis
""",
},
# "Metastases to movable ipsilateral level I and/or level II axillary nodes",
"cN1": {
"Notes": """
Metastases to movable ipsilateral level I and/or level II axillary nodes
These are tumors where CLINICALLY, that is, by palpation, movable lymph nodes are swolled and most likely metastastic
""",
"Relevant Vocabulary": """
Ipsilateral
Axillary
Level I
Level II
Movable
""",
"Sample Sentences": """
LYMPH NODES: 3 sentinel lymph nodes examined with 3 sentinel lymph nodes positive for involvement by macrometastasis
EXTRANODAL EXTENSION Not identified.
LEFT AXILLARY SENTINEL LYMPH NODE, EXCISION: - METASTATIC BREAST CARCINOMA INVOLVING ONE OF TWO LYMPH NODES.
AXILLARY LYMPH NODES: One of two nodes positive (see part 4).
SIZE OF METASTASIS: 2.5 cm
EXTRA CAPSULAR EXTENSION: Present.
LYMPH NODE SAMPLING:
NUMBER OF LYMPH NODES POSITIVE: 2, macrometastasis with extranodal extension.
""",
},
# "Micrometastases",
"cN1mi": {
"Notes": """
Micrometastases
These are very small metastases that can be suspected by clinical examination
""",
"Relevant Vocabulary": """
Micrometastasis
Isolated
Isolated metastatic
""",
"Sample Sentences": """
-1 Sentinel lymph node with isolated tumor cells
Lymph node, sentinel, left axillary #1: -One lymph node, positive for isolated tumor cells by immunohistochemistry.
Lymph node involvement: -Number of lymph nodes with isolated tumor cells (less than or equal to 0.2 mm and less than or equal to 200 cells): 1
""",
},
# "Metastases to fixed or matted ipsilateral level I and/or level II axillary nodes",
"cN2a": {
"Notes": """
Metastases to fixed or matted ipsilateral level I and/or level II axillary nodes
These are tumors whereby CLINICAL examination shows lymph nodes that are fixed and not movable
""",
"Relevant Vocabulary": """
Fixed
Matted
Not movable
Unmovable
""",
"Sample Sentences": """
LYMPH NODES: 4 of 8 lymph nodes positive for metastatic carcinoma (3 macrometastases, 1 micrometastasis and no identified extranodal extension).
After fixation in Dissect-aid, multiple possible lymph node candidates have been identified and submitted as follows:
The specimen was placed in fixative at 15:19.
The specimen was run on the 4:30PM processor.
This tissue was fixed in neutral buffered formalin between 6 and 48 hours.
A second attempt at a lymph node search was completed and few possible lymph nodes were identified and submitted in x!
LYMPH NODES: 5 OF 12 LYMPH NODES POSITIVE FOR METASTATIC CARCINOMA (MACRO-\nMETASTASES WITH EXTRANODAL EXTENSION).

""",
},
# "Metastases to ipsilateral internal mammary nodes without axillary metastases",
"cN2b": {
"Notes": """
Metastases to ipsilateral internal mammary nodes without axillary metastases
These are tumors that metastasize to regional lymph nodes by the internal mammary artery but NOT in the axilla
""",
"Relevant Vocabulary": """
Mammary artery
Mammary nodes
Ipsilateral
""",

"Sample Sentences": """
Internal mammary lymph nodes are involved by metastasis with absence of axillary involvement
Ipsilateral mammary lymph nodes are involved without axillary node involvement
Presence of ipsilateral mammary node metastases with no axillary involvement
Mammary lymph node metastases. No axillary node involvement

""",
},
# "Metastases to ipsilateral level III axillary nodes with or without level I and/or level II axillary metastases",
"cN3a": {
"Notes": """
Metastases to ipsilateral level III axillary nodes with or without level I and/or level II axillary metastases
These are tumors with level III axillary node metastasis regardless of level I or II
""",
"Relevant Vocabulary": """
Level III 
""",
"Sample Sentences": """
Level III axillary lymph nodes are involved
Axillary level III node involvement
Axillary level III metastastic nodes
Metastatic involvement of level III axillary nodes
""",
},
# "Metastases to ipsilateral internal mammary nodes with level I and/or level II axillary metastases",
"cN3b": {
"Notes": """
Metastases to ipsilateral internal mammary nodes with level I and/or level II axillary metastases
These are tumors that combine internal mammary nodes AND axillary level I and/or II
""",
"Relevant Vocabulary": """
Internal mammary AND axillary
Axillary AND internal mammary
""",

"Sample Sentences": """
Internal mammary nodes and level I axillary nodes are involved
Metastatic involvement of internal mammary and level I axillary nodes
Level I and internal mammary axillary node metastases
Axillary level II and internal mammary node metastases
Metastatic involvement of internal mammary and axillary level II nodes
Internal mammary and axillary level II lymph nodes show evidence of metastasis
Positive for metastasis in internal mammary and axillary level II lymph nodes

""",
},
# "Metastases to ipsilateral supraclavicular nodes"
"cN3c": {
"Notes": """
Metastases to ipsilateral supraclavicular nodes
These are tumors with metastasis to the supra-clavicular nodes
""",
"Relevant Vocabulary": """
Supraclavicular 
""",
"Sample Sentences": """
Supraclavicular nodes are involved
Metastatic spread to supraclaviular nodes
Positive for supraclavicular node metastasis
Supraclavicular metastatic nodes are noted
Palpable supraclavicular nodes are involved
"""
}
}

