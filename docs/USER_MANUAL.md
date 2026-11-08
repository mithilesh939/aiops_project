# QuizPulse - User manual (for non-technical users)

QuizPulse tells a teacher or student **which topics are weak** and **what to practise next**, based on past quiz answers.

## Opening the application
1. Ask your administrator to start the system (it runs with one command, `docker compose up`).
2. Open a web browser and go to **http://localhost:8080** (or the address your administrator gives you).
3. Look at the dot in the top-right corner: **green = connected**, **red = the system is not running** (tell your administrator).

## Tab 1 - My learning
1. Type or choose a **Student ID** (for example `S0001`; suggestions appear as you type) and press **Show my topics**.
2. The **Topic mastery** list shows one bar per topic. The longer the bar, the more likely the student is to answer the next question of that topic correctly.
   * Green = strong, amber = in progress, red with the label **WEAK** = below 50%. The thin vertical line marks 50%.
   * Hover over a topic name to see how many past answers the student has and the share that were correct.
3. Press **Suggest 5 questions** to get practice questions from the weakest topics, each with the expected chance of success.
4. **Try a question** (optional): pick a topic and a question, press **Predict** to see the chance of a correct answer as a coloured bar.
   Afterwards press **Correct** or **Wrong** to record what really happened - this makes the system smarter over time.

Tips and messages
* "Cannot reach the QuizPulse API" - the backend is not running; ask the administrator.
* A student with no history still works: topics start around 50% and move as answers arrive.
* The page can be used on a phone; the layout adapts to the screen width.

## Tab 2 - Pipeline & models (for administrators)
* **Model in use** - which trained model version answers requests.
* **Live AUC** - how well recent predictions matched what students really answered (higher is better, ~0.75 is normal for this data).
* **Drift status** - *stable* means the incoming data looks like the training data; *DRIFT* means it changed (for example a new syllabus topic) and the model may become less accurate.
* **Predictions / labels** - number of predictions made and answers received.
* **Run drift check** - compares recent data with the training data now.
* **Retrain now** - trains a new model and switches to it **only if it is better**; otherwise the current model stays.
* **Reload production model** - re-reads which version is "production".
* **Drift report / MLflow / Grafana / Prometheus** - open the detailed tools in a new tab.
* **Pipeline at a glance** and **Recent pipeline runs** - a log of every data, drift and retraining run with its status (green = success, red = failed, orange = skipped). The page refreshes every 5 seconds ("live").

## Tab 3 - Help
A short version of this manual inside the application.

## Frequently asked questions
**What does "weak" mean?** The system estimates less than a 50% chance of a correct answer.
**Can the system be wrong?** Yes - it is a statistical estimate, shown as a percentage. It is a guide for practice, not a grade.
**Is personal data used?** Only student IDs and quiz answers; no names. Data stays on the local computer.
