# Step 0: Validation and minor fixes

- Need to validate the model on Kumpula botanical garden data! Let's hope it's not shit.

## Step 1: Identify heatwaves H1, H2, H3

- We are interested in the hottest hour, of the hottest day, of the heatwave event.
- We adhere to the CTX90 heatwave definition from the paper Juha sent, in which a heatwave means 3 consecutive days where Tmax > 90th percentile of Tmax baseline (1961-1990).
- I download ERA5-Land daily Tmax for May-September 1961-1990 and establish 1 90th percentile for Helsinki.
- Next, I check suspect years until we find our H1, H2, H3. I will collect suspect years from news and press releases, as it will increase relevance.

## Step 2: Predict heatwave + baseline

- Predict H1, H2, H3 max temperature hour. These should no more than a few predictions per event. Easy!
- Predict baseline. This is more complicated:
- For temperature, we could reuse the CTX90 threshold from step 1. I am a bit afraid it might be too conservative for the next step.
- Alternatively, we could use the mean of daily Tmax over June-August 1961-1990 to represent the typical thermal conditions of organisms.
- For the non-temperature variables (SSRD, wind, total precipitation), no idea how to deal with them. Yes, we can also average from baseline here, but it's gonna be 20 gazillion ERA5-Land downloads. At that point, I might as well download the entire dataset.

## Step 3: Answer RQ1

- We take baseline minus our H1-H2-H3 predictions to assess buffering during heatwaves.
- Then we can quantify compare heatwave buffering capacity of different Helsinki green areas through boring descriptive statistics and some nice maps etc. Easy!

## Step 4: Connectivity analysis, answer RQ2

- Senior et al. method! I will steal Iris'es code, if possible.
- We need to be wary of adapting their method assumptions well to our use case. Sensitivity analyses, yippi!
- We run for all three heatwaves and then compare. Hopefully some patterns will emerge and we will be able to identify actual microrefugia as well as thermally isolated and unstable areas.

## Step 4.2: Sensitivity analysis

- Lastly, we just delta shift the temperatures of our baseline up and up until the microrefugia function gets lost.
- And then we compare! Baseline! H1-3! And Baseline+2, +4 degrees or whatever!

I am dreaming of 3-4 really good figures, and a short'ish paper with a punchy headline here.
