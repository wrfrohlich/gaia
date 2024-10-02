# GAIA - Gait Assessment Integration Analyzer


## Effects of additional load at different heights on gait initiation: A statistical parametric mapping of center of pressure and center of mass behavior (MATLAB)

* Motion capture system software - Vicon Nexus, version 2.11, Vicon Motion Systems Ltd, Oxford, UK:
    * Center of Pressure (COP)
    * Ground Reaction Forces (GRF)
    * Kinematic time series

* Filtering: The raw force platform and kinematic data underwent filtering using a fourth-order, zero-lag, low-pass Butterworth filter;
* Filtering: The cutoff frequency for the filter was set at 12.5 Hz [22];
* Data Export: COP, GRF, and kinematics time series were exported in a txt file format;
* COP Calculation: A full description of how to calculate the COP can be found elsewhere [16,23];
* COP Normalization: The COP time series were normalized at the beginning of the task;
* Projection of Center of Mass (COM): This estimation was performed by double integrating the previously exported GRFs using the trapezoidal method;
* APA Phase Identification: The Anticipatory Postural Adjustment (APA) phase was identified according to the method by Vieira et al. [6,16,24].
* Vector Analysis: A vector analysis of the resulting COP and the estimated COM during the APA phase was conducted using the SPM (Statistical Parametric Mapping) method [25];
* Vector Analysis: SPM is a statistical approach that captures features of the entire time series, rather than discrete variables, providing additional information for gait initiation analyses;
* SPM Analysis: SPM analysis uses random field theory to identify field regions that co-vary with the experimental protocol [26,27];
* Data Interpolation: Each Anteroposterior (AP) and Mediolateral (ML) component of each APA time series (COP and COM) was interpolated to contain 61 points;
* Data Interpolation: These 61 points corresponded to approximately 0–30% of the entire task;
* Data Organization: For each APA time series, the AP and ML components were organized into an array;
* Data Organization: Two corresponding matrices were created, each containing 68 rows (one for each subject) and 61 columns;
* Data Organization: This array and matrix organization was done for each experimental condition.
* Statistical Testing: The paired Hotelling’s T-square test, which is the SPM vector field analog to the paired t-test, was used to compare the experimental conditions;
* Statistical Testing: Paired t-tests were conducted as a post-hoc test, with a Sida´k correction to determine significance;
* SPM Output Analysis: The output of the SPM analysis provided the following values for each sample of the COP and COM time series:
    * T-square values
    * t values
    * A threshold corresponding to α (alpha) set at 0.05.
* Identification of Significant Differences: In the output, values of T-square and t that were above the threshold indicated significant differences in the corresponding portion of the Anticipatory Postural Adjustment (APA) time series.
* SPM Analysis Tools: To conduct the SPM analysis, Matlab codes provided by www.spm1d.org were used.

## Margins of stability of persons with transtibial or transfemoral amputations walking on sloped surfaces (MATLAB)

* Data Filtering: Before data analysis, the kinematic data underwent low-pass filtering using a fourth-order, zero-lag Butterworth filter;
* Data Filtering: The cutoff frequency for the filter was set at 8 Hz;
* Data Preprocessing: The initial and final 15 seconds of each trial were discarded (Hak et al., 2013a);
* Data Preprocessing: Steps were detected based on the zero-cross of the heel-marker velocity in the Anteroposterior (AP) direction (Souza et al., 2017);
* Data Preprocessing: Only the intermediate 150 strides were selected, and any initial and final strides exceeding 150 were excluded;
* Data Analysis Tools: The data analysis was performed using a custom MATLAB code;
* Computation of Spatiotemporal Parameters: To interpret the Margin of Stability (MoS) results, related spatiotemporal parameters were computed, including:
    * Step Frequency (SF), determined as the inverse of the average duration of the steps;
    * Average Step Length (SL), computed from the average step frequency and average treadmill speed;
    * Walk Ratio (WR), calculated as SL normalized by SF;
    * Step Width (SW), determined as the Mediolateral (ML) distance between heel markers within two subsequent heel strikes;
* Gait Stability Assessment: Gait stability was assessed using the Margin of Stability (MoS);
* Gait Stability Assessment: MoS was estimated based on the method proposed in Hak et al., 2013a;
* MoS Calculation for Prosthetic and Intact Limb: MoS for prosthetic and intact limbs was calculated separately;
* Statistical Analysis: Since the variables had normal distributions (Shapiro-Wilk test, p > 0.05), mixed repeated-measures analysis of variance (mixed ANOVA) was used to assess:
    * Main effects of inclination and groups (and limb for MoS);
    * Interaction effect between group and inclination;
    * A post hoc test with Bonferroni correction was performed when a main or interaction effect was significant;
    * Paired T-tests were conducted to compare MoS between the prosthetic and intact limbs;

## Biomechanical responses of Nordic walking in people with Parkinson's disease (Matlab)

* Spatiotemporal Parameters: The following spatiotemporal parameters were calculated:
    * Stride Length (SL): Obtained by multiplying the forward velocity (Vf) by the period of stride (T) (SL = Vf * T).
    * Stride Frequency (SF): Obtained using the inverse of the stride period (SF = 1/T).
    * Single Support: Calculated as the fraction of the period during which only one foot contacts the ground.
    * Double Support: Calculated as the fraction of the period during which both feet contact the ground.
    * Lateral and Vertical Oscillations: Calculated as the difference between the maximum and minimum positions in both directions of the center of body mass.

* Statistical Analysis: Data are presented as the mean and standard deviation;
*  Statistical Analysis: For individual sample characteristics, data were compared using an independent-sample t-test;
*  Statistical Analysis: A generalized linear model was used to identify the main effects of group (control vs. Parkinson), modality (FW vs. NW), and group × modality interactions;
* Statistical Analysis: Statistical analysis was performed using the Statistical Package for Social Sciences (SPSS) version 26, with a significance level set at α = 0.05.

## Barefoot walking changed relative timing during the support phase but not ground reaction forces in children when compared to different footwear conditions (Python)

* SMART-ANALYZER (BTS Bioengineering Corp., Quincy, MA, USA)
* Data Reduction: Python 3 was used to visually inspect and detect force signals and their timing in each gait cycle;
* Filtering: The software filtered the raw data using a 4th order low-pass Butterworth zero-lag filter with a cut-off frequency of 20 Hz;
* Calculated: It calculated various relevant parameters for each step (e.g., F1, F2, F3, F4, F5, tc, t1, t2, t3, t4, t5, t6, t7, I1, I2, I3, I4, and I5).
* Data process: Steps with force curves that didn't show the characteristic two peaks and valley of walking gait were manually removed from the analysis.
* Data process: The means of the relevant parameters were calculated by averaging their values across the remaining valid steps.
* Data process: Time data were normalized to the length of the support phase (%tc), force data were normalized by the participants' body weight (BW), and impulse data were calculated using the normalized force (BW*s).
* Statistical Analysis: A repeated-measures ANOVA was performed to investigate the effects of the footwear type on gait parameters in each group
Mauchly’s test of sphericity was conducted to assess the assumption of homogeneity of variance.
* Statistical Analysis: When sphericity was violated, the degrees of freedom of the omnibus F-test were corrected using Greenhouse-Geisser estimates of sphericity (ε).
* Statistical Analysis: Multiple comparisons with Bonferroni corrections were conducted when the omnibus F-test was significant.
* Statistical Analysis: Statistical analyses were performed in SPSS v25.0 (IBM, Armonk, NY, USA).
* Statistical Analysis: The level of significance was set a priori at α = .05 for all tests.

## Paper 1:

* Gait Protocol: The study involved a gait protocol with different footwear models and barefoot gait conditions.

* Data Collection: Participants completed trials walking across force platforms.

* Data Reduction: Python and SMART-ANALYZER software were used to select and process gait data.

* Data Processing: Parameters like force and timing were calculated, and data were normalized.

* Statistical Analysis: Repeated-measures ANOVA was used to assess the effects of footwear on gait parameters.

## Paper 2:

* Motion Capture Data Acquisition: The study used motion capture technology to collect COP, GRF, and kinematic data.

* Data Filtering: Data underwent low-pass filtering using a Butterworth filter.

* Data Export: COP, GRF, and kinematic data were exported in txt files.

* Data Normalization: COP time series were normalized, and COM projection on the force platform was estimated.

* APA Phase Identification: Anticipatory Postural Adjustment (APA) phases were identified.

* Vector Analysis: SPM method was used for vector analysis of COP and COM during APA.

* Data Interpolation: Time series data were interpolated to contain 61 points.

* Statistical Testing: Statistical analysis included Hotelling’s T-square test and paired t-tests.

## Paper 3:

* Gait Protocol: The paper mentions a gait protocol but does not provide specific details.

* Data Collection: Data acquisition of mechanical parameters is mentioned.

* Data Reduction: Specific data reduction methods are not detailed in the text.

* Data Processing: Normalization of time and force data is mentioned.

* Statistical Analysis: Repeated-measures ANOVA and post hoc tests were conducted to analyze gait parameters.

## Paper 4:

* Gait Protocol: The study involved gait analysis under various conditions.

* Data Collection: Specifics of data collection are not detailed in the text.

* Data Reduction: The paper discusses data reduction without providing specific methods.

* Data Processing: Specific data processing methods, including normalization, are not detailed.

* Statistical Analysis: Repeated-measures ANOVA and Bonferroni corrections were used for statistical analysis.
