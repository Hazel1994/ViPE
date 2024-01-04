# LyricCanvas Dataset
- The lyricCanvas dataset contains approximately 10M lines of lyrics with corresponding visual elaborations (visualizable prompts).
- It could be used to train large language models to translate highly abstract concepts and metaphorical
phrases to visualizable prompts for image generation
- Due to copy right policies, we are not allowed to publish the lyrics, however, we release the visual elaborations and the scraper through which
you can collect the lyrics and rebuild LyricCanvas with no additional cost.

## Compiling LyricCanvas
Building LyricCanvas involves two main steps:
- Step 1: Scraping lyrics from the Genius platform followed by a preprocessing step
- Step 2: Generating synthetic visual elaboration using a large language models (e.g., GPT3.5 Turbo) followed by a post-processing step

### Step 1: Scraping Genius Lyrics
The first part of creating the LyricCanvas dataset is to scrape the lyrics form the
genius platform

Run the `scraper.py` script with the following arguments to scrape lyrics:

```bash
python scraper.py --start <start_index> --end <end_index> --path <output_path> --genius_token <your_genius_token>
```
- `--start`: Optional. Start index for scraping artists, default=0.
- `--end`: Optional. End index for scraping artists, default=-1.
- `--path`: Required. A directory, where the scraped lyrics will be saved.
- `--genius_token`: Required. Your Genius API token.

**Note:** You might consider running multiple jobs in parallel using different *--start* and *--end* values.

After scraping the lyrics, preprocess them and save them as a single dataset using the prepare_lyrics.py script:

```bash
python prepare_lyrics.py --path <scraped_data_path> --path_out <output_path>
```
- `--path`: Required. Path where the scraped lyrics are saved.
- `--path_out`: Required. Path to save the prepared lyrics.
See other options using `--help`\
The resulting dataset is saved as a dictionary file with the following structure:
```python
dict = {
    "artist_name": [
        { "title": "song1", "lyrics": "l1", ... },
        { "title": "song2", "lyrics": "l2", ... }
    ],
    ...
}
```
#### Viewing Dataset Statistics
The info.py script provides statistics about the prepared dataset:
```bash
python info.py --path_out <prepared_data_path>
```
- `--path_out`: Path to the prepared lyrics dataset.
---
**Our version of the genius lyrics contain the following statistics :**
- number of artists:  5549
- total songs:  249948
- total lines:  9909617
- mean , std  lines per track:  39.646714516619454   10.77...
- min , max  lines per track:  14   50
- number of bad name or title 0
---

### Step 2: Generating Visual Elaborations
Once you have the lyrics prepared, you could either:
- Option 1: Use our generated visual elaborations available [here](https://huggingface.co/datasets/fittar/lyric_canvas).
- Option 2: Generate new visual elaborations by using an LLM

### Option 1: Use our generated visual elaborations 
For our set of lyrics, the prompts from GPT3.5-Turbo are available [here](https://huggingface.co/datasets/fittar/lyric_canvas). Simply download the file and use the
'build_lyric_canvas.py' script to build the complete version of the dataset as follows

```bash
python build_lyric_canvas.py ---lyric_file <output of prepare_lyrics.py> --ds_file <final dataset file>
--prompt_path <generated prompts> --vipe_prompts <Use our provided prompts>
```
- `--lyric_file`: path to the pickle file from prepare_lyrics.py, including the name of the pickle file
- `--ds_file`: path to save the final version of lyricCanvas including the name, e.g., /a/b/lyric_canvas_complete.csv
- `--prompt_path`: either a directory where ChatGPT saved the prompts or a path to our generated prpmpts (the csv file)
- `--vipe_prompts`: pass this flag if you like to use our provided prompts and skip using ChatGPT


The resulting Dataset is a **csv** with the lyrics column filled with the prepared lyrics.

### Option 2: Generate new visual elaborations by using an LLM
using the 'chatgpt_generate.py' script, you can utilize ChatGPT with our system role to generate prompts for the scraped lyrics.

```bash
python chatgpt_generate.py ---path_data_input <output of prepare_lyrics.py> --path_data_output <where to save the prompts>
--path_log_output <where to save the logs> --api_key <your OpenAI API Key> --n_chunks <number of calls in parallel>
```
- `--n_chunks`: Divide the lyrics into this many chunks and for each chunks run a separate thread for calling ChatGPT, Make sure
you dont run out of your token/call quota per min/hour

Once the prompts are generated, us the `build_lyric_canvas` without passing the `--vipe_prompts` to build the final dataset

