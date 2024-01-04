# User Study: Comparative Evaluation of Metaphor Representations


## Methodology
### Data Preparation
- **Dataset:** The HAIVMet dataset was used.
- **Metaphor Selection:** 60 metaphors were randomly selected.
- **Visual Elaborations:** For each metaphor, visual elaborations were generated using ChatGPT, ViPE, and human expert inputs from HAIVMet.
- **Image Generation:** Stable Diffusion was employed to generate corresponding images from these visual elaborations.

### Experiment
- Participants were presented with a metaphor alongside three images.
- Images were generated from prompts provided by human experts (HAIVMet dataset), ChatGPT, and ViPE.
- Participants were asked to select the image that best represented the metaphor's meaning.
- 3 samples with obvious solutions were added for sanity checks
## Findings
- **Preference Percentage:**
    - Human Experts: 38.67%
    - ViPE: 33.61%
    - ChatGPT: 27.72%
- The results suggest a preference for images from human experts, followed by ViPE and ChatGPT, respectively.
- These findings validate ViPE's superiority over ChatGPT and its competitive performance with human experts.

## Link to our User Study
[Metaphorical Expression and Visual Perception](https://forms.gle/jM2dcLUUyn648HGA9)

## replicating the results
Simply run the `analyse.py` with the results of the user study in the current directory