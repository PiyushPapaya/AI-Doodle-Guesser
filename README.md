# AI-Doodle-Guesser
A ML project that predicts what a user is drawing using a CNN trained on the Google "Quick, Draw!" dataset.

## How to Get Started

You can run this project easily in **Kaggle** using a free GPU (T4 ×2). Here’s what to do:

### 1. Open the notebook in Kaggle

- Set the accelerator to **GPU (T4 ×2)** under “Settings”.

### 2. Download the model & labels

- Download the latest trained model (`model.h5`) and `labels.txt`.
- Move them into a folder called `sketchmodel/`.
  - You should now have:
    - `sketchmodel/model.h5`
    - `sketchmodel/labels.txt`

You can change the labels or add more classes if you want!\
Also feel free to play around with the **learning rate**, **batch size**, and the **number of classes** when training.

### 3. Install the required libraries

In the your coding environment, upload the folder and run:

```bash
pip install -r requirements.txt
```

### 4. Edit the UI (optional)

Open `index.html` and change the layout, title, or design to match your style.

You can also add more buttons, tools, or sketch options.

### 5. Run the app

Start the app locally:

```bash
python app.py
```

Then go to `http://127.0.0.1:5000` in your browser and start drawing!

## How It Works

- The app takes your sketch stroke-by-stroke.
- It sends the image to a neural network trained on your selected classes.
- The model returns the top prediction in real-time.

## Tips

- Make sure your model and labels match.
- Keep testing with new drawings to improve accuracy.
- Try training with more classes or improving the UI!

## Files & Folders

```
sketchmodel/
  ├── model.h5
  ├── labels.txt
  └──static/
        └── index.html
  ├──app.py
  └──requirements.txt
```

## Enjoy Drawing!

This is a project to explore sketch recognition and real-time AI predictions. Have fun and feel free to experiment!
