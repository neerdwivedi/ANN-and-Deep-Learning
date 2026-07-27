# Email-Style Message Sorter — Keras MLP for Multiclass Classification

For this practical I built a program that reads a piece of text and decides which topic it
belongs to — the same basic idea behind Gmail sorting mail into tabs. In this README I explain
what I did, why I made each choice, and what other options I could have used instead. I've tried
to keep it in plain words so it's easy to follow.

---

## 1. My aim

My goal was to make a program that can **read a message and sort it into one of four topic boxes**:

- **Tech** (computer graphics talk)
- **For sale** (people selling things)
- **Sports** (ice hockey talk)
- **Science** (space talk)

I don't give the computer any fixed rules like "if it says NASA it's Science." Instead I show it
thousands of real messages that already have their topic attached, and it learns the patterns on
its own. After that, when I give it a brand-new message it has never seen, it makes a smart guess
about which box it belongs in.

This kind of "sort into one of several boxes" task is called **multiclass classification** —
"multi" because there are more than two boxes, and "classification" because I'm predicting a
category rather than a number.

---

## 2. Where this topic fits

To understand what I'm doing, it helps to see the family tree:

- **Artificial Intelligence (AI)** — the broad idea of making computers act smart.
- **Machine Learning (ML)** — a part of AI where the computer learns from examples instead of
  being given hand-written rules. My project sits here.
- **Neural Networks** — one popular ML method, loosely inspired by how brain cells connect.
- **MLP (Multi-Layer Perceptron)** — the specific, simple type of neural network I built.

I'm also doing **supervised learning**, which just means my training data comes with the correct
answers already attached (every message has its topic label). The model learns by comparing its
guesses to those known answers and correcting itself.

So the full path of my project is: **AI → Machine Learning → Neural Network → MLP**, doing
**supervised multiclass classification**.

---

## 3. The dataset I used

I used the well-known **20 Newsgroups** dataset. In the early days of the internet, "newsgroups"
were public discussion boards (like old forums), each about a topic. This dataset is a big pile of
real messages people posted, already grouped by which board they came from — so every message is
pre-labelled with its topic, which is exactly what supervised learning needs.

The full dataset has 20 topics, but I only used **4** to keep things simple:

| File on disk | Original topic | Name I gave it |
|---|---|---|
| `comp.graphics.txt` | computer graphics | Tech |
| `misc.forsale.txt` | items for sale | For sale |
| `rec.sport.hockey.txt` | ice hockey | Sports |
| `sci.space.txt` | space & astronomy | Science |

Each file holds around **1,900 messages** stacked one after another. Every message starts with a
line beginning `Newsgroup:`, which is how my code knows where one message ends and the next begins.

**Why I chose these four:** they're very different topics, so the words barely overlap (space talk
looks nothing like hockey talk). That makes the job easier and lets me reach high accuracy, which
is good for a first project. If I'd picked four similar topics (like three different politics
boards), the words would overlap a lot and the model would find it much harder — that's something
I could try later as a tougher experiment.

---

## 4. The pipeline (my whole process)

Every classification project follows the same recipe, and mine is:

1. **Load** the messages from the files.
2. **Turn the words into numbers** (the one step that's special to text).
3. **Split** the data into a training set and a testing set.
4. **Build** the neural network.
5. **Train** it on the training set.
6. **Test** it on the unseen testing set and judge how it did.
7. **Use** it on a brand-new message.

The rest of this README goes through each step and explains why I did it that way, why that's a
good choice, and what the alternatives were.

---

## 5. Step by step — my choices and the alternatives

### Step 1 — Loading the messages

**What I did:** read each topic file and split it into individual messages wherever a `Newsgroup:`
line appears. I also **remove the header block** at the top of each message (the `From:`,
`Subject:`, `Organization:` lines).

**Why I remove the headers:** because they'd let the model cheat. The header often says which board
the message came from, so the model could just read the header instead of understanding the actual
text. Removing it forces the model to learn from the real message content, which is the honest way.

---

### Step 2 — Turning words into numbers (the key idea for text)

A neural network can only do maths on numbers, but a message is words. So before anything else I
have to turn each message into a row of numbers. I used a method called **TF-IDF**.

**What TF-IDF does, in plain terms:** it looks at all the messages, picks the 2,000 most useful
words, and describes each message by how strongly each of those words shows up in it. Words that
appear a lot in one message but rarely elsewhere (like "rocket" or "goalie") get a high score,
because they're the ones that really tell topics apart. Filler words that appear everywhere (like
"the" and "and") get thrown away — those are called stop words.

So a message becomes 2,000 numbers, just like an Iris flower is 4 numbers. Same idea, bigger.

**Why I chose TF-IDF, and the alternatives:**

- **Plain word counting (Bag of Words):** just counts how many times each word appears. It's
  simpler, but it treats common words as important just because they're frequent. TF-IDF is
  smarter because it lowers the score of words that appear everywhere and boosts the distinctive
  ones. It's basically Bag of Words with a fairness adjustment, so I preferred it.
- **Word embeddings / modern language models (Word2Vec, BERT):** these actually understand meaning
  and word order, so they're far more powerful. But they're heavier, slower, and much harder to
  set up. For my project, where the topics are clearly different, that power is overkill — TF-IDF
  already gets near-perfect accuracy, so the extra complexity isn't worth it. Those tools are worth
  it when topics are subtle or the exact wording matters.

**In short:** TF-IDF is the sweet spot — smarter than plain counting, far simpler than the heavy
methods, and more than good enough here.

---

### Step 3 — Splitting into training and testing sets

**What I did:** kept 80% of the messages for the model to learn from, and hid the other 20% away
to test it afterwards.

**Why:** otherwise I couldn't tell real learning apart from memorising. If I tested the model on the
same messages it studied, of course it would score high — it already saw the answers. That's like
giving a student the exam questions to memorise beforehand; a high score wouldn't prove anything.
Testing on messages it has never seen is the only fair measure.

I also used `stratify`, which just makes sure all four topics appear evenly in both the training and
testing piles, so no topic gets left out of the test.

**Alternative:** a method called **cross-validation** splits the data many ways and averages the
results, which is even more reliable. It's the proper method for serious work, but it's slower and
more than a first project needs. A single 80/20 split is the standard, simple choice for learning.

---

### Step 4 — Building the neural network (the MLP)

My network has three layers stacked up:

```
Input (2000 numbers)  ->  Hidden layer (64 neurons, ReLU)  ->  Output (4 neurons, softmax)
```

**Why I have a hidden layer:** the hidden layer is what lets the network learn *combinations* of
clues instead of single words. Without it, the model can only make simple straight-line rules; with
it, it can combine evidence ("mentions engines AND selling AND a price") to make smarter decisions.
That middle layer is literally what the "multi-layer" in MLP means.

**Why ReLU on the hidden layer:** ReLU is the little rule each neuron uses to decide whether to
"fire." It's the default in modern networks because it's simple and makes training fast and stable.
Older options (sigmoid, tanh) tend to slow training down, so ReLU has become the go-to.

**Why softmax on the output, and why 4 neurons:** four neurons because I have four topics — one per
box. Softmax turns their raw scores into percentages that add up to 100%, so the four boxes compete
and exactly one wins ("92% Science, 5% Tech, 2% Sports, 1% For sale").

- **Alternative — sigmoid on each output:** sigmoid is the tool for yes/no problems. If I put it on
  all four outputs, each would answer on its own, and nothing would stop all four saying "90%",
  which makes no sense when only one box is right. Softmax is the correct choice because it forces
  the four to trade off against each other. Sigmoid is for binary; softmax is for multiclass.

**Why Keras (TensorFlow):** Keras lets me build a neural network in a few readable lines instead of
writing all the maths by hand. It's one of the most popular, beginner-friendly tools, and it's what
the practical asked for.

- **Alternative — scikit-learn's `MLPClassifier`:** builds the same kind of network with even less
  code and installs more easily, but gives less control and is less standard for deep-learning
  coursework. **PyTorch** is another big option — more flexible and popular in research, but harder
  for beginners. Keras is the friendly middle ground, so I used it.

---

### Step 5 — Training

**What I did:** let the model look at the training messages 15 times over (each full pass is called
an **epoch**), adjusting itself a little each time to make fewer mistakes. I used an **optimizer**
called **Adam**, which decides how big those adjustments should be.

**Why Adam:** it automatically tunes the size of each learning step, so it trains quickly and
reliably without much fiddling. It's the sensible default most people start with.

- **Alternative — plain SGD (Stochastic Gradient Descent):** the classic, simpler optimizer. It
  works, but usually needs hand-tuning to train as fast as Adam. Adam is basically a smarter,
  self-adjusting version of it, which is why I picked it.

**The loss function — categorical crossentropy.** "Loss" is a score of how wrong the model is;
training tries to make it as small as possible. Crossentropy looks at the percentage the model gave
to the correct answer and punishes it harder the lower that was — and it punishes being confidently
wrong the most.

- **Alternative — squared error (MSE):** the loss used for predicting numbers. It works poorly for
  classification because it barely reacts to confident mistakes. Crossentropy is made for "pick a
  category" problems, so it's the right partner for softmax.

(I use the *sparse* version, `sparse_categorical_crossentropy`, just so my labels can stay as simple
numbers 0–3 instead of a more awkward format. Same maths.)

---

### The specific numbers (settings) I chose, and why

These small numbers are called **hyperparameters** — settings I pick *before* training, as opposed
to the things the model learns by itself. They matter, so here's each one and my reasoning.

**Epochs = 15.** An **epoch** is one full pass where the model looks at every training message once.
More epochs means the model studies the data more times and usually improves — up to a point. If I
use too *few*, the model hasn't learned enough (called *underfitting*). If I use too *many*, it
starts memorising the training data instead of learning the general pattern, and gets worse on new
data (*overfitting*). I picked 15 because my topics are easy to tell apart, so the model already
reaches near its best within a handful of passes; going higher just wastes time and risks
overfitting. If accuracy were still climbing at the end, that would be a sign to raise it.

**Batch size = 32.** The model doesn't look at all messages at once — it works through them in small
groups called **batches**, updating itself after each group. A batch size of 32 means it adjusts
after every 32 messages. Smaller batches update more often (slower but sometimes more accurate);
larger batches are faster but coarser. 32 is the common default that balances speed and stability,
so I used it.

**Number of words kept (max_features) = 2000.** In Step 2, TF-IDF keeps only the 2,000 most useful
words. I could keep more, but most extra words are rare and add little while making the model bigger
and slower. 2,000 is plenty to separate four clear topics, so it's a good speed-vs-accuracy trade.

**Hidden layer size = 64 neurons.** This is how much "thinking room" the middle layer has. Too few
neurons and the model can't capture enough patterns; too many and it's slow and can overfit. 64 is a
reasonable middle value for a problem this size — big enough to do the job, small enough to stay fast.

**Validation split = 0.1.** During training I set aside 10% of the *training* data as a quick check
(a **validation set**) to watch how the model does on data it isn't directly learning from. It's an
early-warning system for overfitting, and it's separate from the real test set, which I keep
untouched until the very end.

None of these are magic numbers — they're sensible defaults. A normal next step would be to try a
few values of each and see which does best, but for a first working project these choices are solid.

---

### Step 6 — Testing and judging the results

Accuracy alone doesn't tell the whole story, so I look at three things:

- **Accuracy** — the overall percentage it got right. The simple headline number.
- **Precision, recall and F1 (the per-class report)** — these break the score down per topic, so I
  can spot if the model is great at some boxes and weak at others. "Recall" for Sports, for example,
  means: of all the messages that really were Sports, how many did it catch?
- **Confusion matrix** — a grid showing exactly which mistakes it made (like how many Tech messages
  it wrongly called For sale). The correct answers sit on the diagonal; anything off the diagonal is
  a mix-up. This is the most useful view, because the type of mistake often makes sense (a message
  selling a graphics card could look like both Tech and For sale).

**Why I don't just trust accuracy:** with uneven data, accuracy can hide problems. A model could
score high overall while completely failing one small category. The per-class report and confusion
matrix show that; a single number doesn't.

---

### Step 7 — Using it on a new message

Finally I hand the trained model a fresh sentence it has never seen, run it through the same
word-to-number step, and let it predict. This is the real payoff: it shows the model learned a
general pattern instead of just memorising the training data.

---

## 6. Why my approach is a good one (and its limits)

**Why it's a solid choice:** the pipeline (load → numbers → split → build → train → test) is the
standard way to do text classification. Every tool I picked is the sensible default for a beginner:
TF-IDF for text, an MLP in Keras for the model, Adam and crossentropy for training. Nothing exotic,
everything explainable — which is what I wanted while learning.

**Its limits (being honest):**

- TF-IDF ignores word order and meaning — it just counts words. "The dog bit the man" and "the man
  bit the dog" look the same to it. For my easy, distinct topics that doesn't matter, but for subtle
  language it would.
- My high 99% accuracy is partly because the four topics are so different. Harder, overlapping
  topics would score lower — and that's normal, not a failure.

**Alternatives to the whole model, and when I'd use them:**

- **Naive Bayes** — a classic, very fast text classifier. Often nearly as accurate as mine on simple
  topic-sorting and much lighter. A great baseline that many people would try first.
- **Logistic Regression / SVM** — simple, strong, reliable classifiers that also work very well on
  TF-IDF text. Often the "just works" choice in industry.
- **Deep language models (BERT and similar)** — the heavyweight option that understands meaning and
  context. I'd reach for these when topics are subtle, wording matters, or I need the best possible
  accuracy — but they need far more complexity and computing power.

I chose the **MLP** because this practical is specifically about neural networks, and it's a nice
balance: more capable than the simplest models, far simpler than the heavy ones, and it teaches the
core neural-network ideas (layers, ReLU, softmax, training) clearly.

---

## 7. How to run it

I need Python installed, then from a terminal (Command Prompt) inside this folder:

```
py -m pip install tensorflow scikit-learn numpy matplotlib
py email_sorter.py
```

It prints the loading progress, trains the network, then shows the accuracy, the per-class report,
the confusion matrix, and a prediction on a sample sentence.

---

## 8. Quick glossary

- **Feature** — one input measurement. Here, the score for one word.
- **Label / class** — the correct answer (which topic box).
- **Supervised learning** — learning from examples that come with the correct answers.
- **TF-IDF** — a way of turning text into numbers that highlights distinctive words.
- **MLP** — a simple neural network made of stacked layers.
- **Neuron** — one tiny decision unit inside a layer.
- **Hidden layer** — a middle layer that lets the network combine clues.
- **ReLU** — the simple on/off rule each hidden neuron uses.
- **Softmax** — turns the final scores into competing percentages that add up to 100%.
- **Loss** — a score of how wrong the model is; training shrinks it.
- **Crossentropy** — the loss used for "pick a category" problems.
- **Optimizer (Adam)** — the part that decides how the model adjusts itself while learning.
- **Epoch** — one full pass through all the training data.
- **Accuracy / precision / recall** — different ways of measuring how well it did.
- **Confusion matrix** — a grid showing which mistakes were made.