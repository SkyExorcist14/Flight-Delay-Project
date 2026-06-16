DATA 603 - Flight Airline Industry Predictive Modeling Project
SkySense Flight Intelligence Platform

Project Overview
----------------
SkySense is a flight intelligence and decision-support platform built for the DATA 603 Big Data Platforms final project. The project uses large-scale historical flight delay, fare, and weather-related datasets to analyze airport performance, identify delay and fare trends, train machine learning models, and present the final insights through an interactive Flask web application.

The complete raw dataset used during development was approximately 21.2GB. Because of file-size limitations, the full raw dataset is not included in the submission package. Instead, the submission includes project source code, processed Spark outputs, model outputs, representative sample data, README instructions, and the presentation deck.

Team Deliverables Included
--------------------------
1. Slides deck:
   - Flight_Project_Presentation.pptx

2. Project source code:
   - Step1_find_airports (top 20).py
   - Step2_delay_analysis.py
   - Step3_fare_analysis.py
   - Step4_weather_analysis.py
   - Step5_ML_Model.py
   - Step5A_delay_model_improved.py
   - Step6_graph_analysis.py
   - app.py
   - webapp.html

3. Sample data:
   - Sample_Data/ contains representative sample files from the delay, fare, and weather datasets.
   - The full raw dataset is not included due to size.
   - The Processed/ folder contains Spark-generated output files used by the web application and presentation.

4. README file:
   - README.txt

5. Web application files:
   - app.py
   - webapp.html
   - results.json

Project Goal
------------
The goal of this project is to analyze the airline industry using historical flight, fare, and weather data. The project focuses on identifying top airports, understanding delay behavior, analyzing fare patterns, evaluating weather-related disruption risk, building machine learning models for prediction, and presenting results through an interactive web application.

The final web app allows users to select an origin airport, destination airport, departure date, trip type, and cabin option. It then displays expected delay, estimated fare, cancellation/delay risk, carrier comparison, route recommendation, and supporting charts. The app automatically derives weekday/weekend behavior from the selected departure date.

Main Project Steps Completed
----------------------------
Step 1 - Find Top Airports
- Built an RDD containing major US airport information.
- Listed more than 20 US airports.
- Calculated historical departure counts, arrival counts, and total airport traffic.
- Identified the busiest airports in the network.

Step 2 - Delay Analysis
- Analyzed departure and arrival delay behavior.
- Identified airports with the highest departure delays.
- Identified airports with the highest arrival delays.
- Calculated busiest airports, monthly delay trends, daily delay trends, and delay-cause summaries.

Step 3 - Fare Analysis
- Analyzed historical fare data by year, quarter, and origin airport.
- Identified cheapest airports, most expensive airports, and best-value airports.
- Calculated average fares, itinerary counts, distance-related fare behavior, and cost-per-mile patterns.

Step 4 - Weather Analysis
- Integrated weather-related delay behavior with airport-level summaries.
- Identified airports most affected by weather delays.
- Calculated average weather delay and severe weather delay percentages.
- Used weather-delay patterns as part of travel-risk analysis.

Step 5 - Machine Learning Model
- Trained delay prediction models using PySpark ML.
- Built a baseline delay model using RandomForestRegressor.
- Built an improved delay model using Gradient Boosted Tree Regressor.
- Trained a fare prediction model using Gradient Boosted Tree Regressor.
- Created an ensemble-style recommendation layer for BUY / CAUTION / AVOID route advice.

Feature Engineering in Step 5
-----------------------------
Feature engineering was performed before training the machine learning models. Raw flight, fare, time, route, and weather-related columns were transformed into model-ready features.

Delay model feature engineering included:
- Time-based features such as month, day of week, and departure hour.
- Travel-pattern flags such as weekend, summer, winter, and holiday-season indicators.
- Encoded categorical variables such as carrier, origin airport, and destination airport.
- Route and airport history features such as origin delay, destination delay, route delay, and carrier-route delay.
- Operational features such as distance and scheduled elapsed time.
- Weather-delay signals to capture weather-related disruption risk.

Fare model feature engineering included:
- Year and quarter-based fare trends.
- Origin airport encoding.
- Distance-related fare behavior.
- Yield-per-mile / cost-per-mile style fare indicators.
- Peak-quarter indicators for seasonal fare patterns.

Step 6 - Graph Analysis and Web Data Generation
- Built route lookup tables and airport network summaries.
- Combined processed outputs into a single results.json file.
- Prepared structured data for charts, airport tables, carrier comparison, fare summaries, route search, and recommendation logic.
- Made the web application lightweight by using precomputed Spark outputs instead of rerunning the full pipeline live.

Step 7 - Web Application
- Built an interactive Flask-based web application.
- app.py serves the dashboard locally.
- webapp.html contains the user interface and visual dashboard.
- results.json provides the processed data used by the dashboard.
- Users can search routes and view delay estimates, fare estimates, carrier comparisons, charts, and recommendations.

How to Run the Web App Locally
------------------------------
Recommended method:

1. Open PowerShell, Command Prompt, or the VS Code terminal.

2. Go to the project folder:

   cd "PATH_TO_PROJECT_FOLDER"

3. Install the basic libraries needed for the web app if they are not already installed:

   python -m pip install flask pandas numpy

4. Start the Flask app:

   python app.py

5. Open this URL in Chrome or another browser:

   http://127.0.0.1:5000

Important:
Do not open webapp.html directly by double-clicking it. Browser security can block results.json when loaded using file://. Use python app.py and open the localhost link instead.

Alternative Local Static Server
-------------------------------
If Flask is not available, the dashboard can also be served as a static page:

   python -m http.server 8000

Then open:

   http://127.0.0.1:8000/webapp.html

How to Re-run the Full Spark Pipeline
-------------------------------------
The raw data is not included in the submission package because it is very large. To fully rerun the analysis scripts, place the original datasets in local folders and update the path variables inside the scripts if needed.

Recommended raw data structure:

Data/
  Delays/
    *.csv
  Fare/
    *.csv
  Weather/
    *.csv

Recommended run order:

1. python "Step1_find_airports (top 20).py"
2. python Step2_delay_analysis.py
3. python Step3_fare_analysis.py
4. python Step4_weather_analysis.py
5. python Step5_ML_Model.py
6. python Step5A_delay_model_improved.py
7. python Step6_graph_analysis.py
8. python app.py

Environment Notes for Spark Scripts
-----------------------------------
The Spark analysis scripts were developed on Windows using:
- Python 3.11
- Java 17
- PySpark 3.5.1
- Hadoop winutils setup for local Windows Spark execution

If running the full Spark pipeline on another machine, the user may need to update local path variables in the scripts, such as:
- PYTHON_EXE
- JAVA_HOME
- HADOOP_HOME
- PROJECT_ROOT
- delay_folder
- fare_folder
- weather_folder
- processed/output folder paths

For rerunning the Spark scripts, the following libraries/tools may be needed:
- pyspark
- pandas
- numpy
- scikit-learn if used by local helper code
- Java 17 or compatible Java version
- Windows Hadoop/winutils setup if running Spark locally on Windows

Processed Outputs
-----------------
The Processed/ folder contains output files generated from the Spark analysis pipeline. These files are used for visualizations, model reporting, and web application data generation.

Examples of processed outputs include:
- busiest airports
- top departure delay airports
- top arrival delay airports
- monthly delay summaries
- daily delay summaries
- delay causes
- fare by year
- fare by quarter
- cheapest airports
- most expensive airports
- best-value airports
- worst weather airports
- model metrics
- route lookup summaries

The web app uses results.json, which was generated from these processed outputs.

Key Model Metrics
-----------------
Delay baseline model:
- Model: RandomForestRegressor
- RMSE: 36.73 minutes
- MAE: 18.08 minutes
- R2: 0.0175

Improved delay model:
- Model: GBTRegressor
- RMSE: 26.76 minutes
- MAE: 14.92 minutes
- R2: 0.1132

Fare model:
- Model: GBTRegressor
- RMSE: $37.15
- MAE: $18.85
- R2: 0.9846

Assumptions
-----------
1. Historical flight, fare, and delay patterns are useful indicators of future travel risk.
2. Route-level and airport-level averages are used when exact future flight information is not available.
3. The web app provides decision support, not guaranteed real-time flight or fare predictions.
4. Large raw datasets are excluded from the submission, while representative sample data and processed outputs are included.
5. Fare estimates are based on historical fare patterns, not live airline pricing APIs.
6. Weather risk is based on historical weather-related delay behavior, not live weather forecasts.
7. The web app uses precomputed results for fast interaction instead of retraining models live.

Known Limitations
-----------------
1. The project does not use live flight status APIs.
2. The project does not use live airline fare APIs.
3. Future travel recommendations are based on historical patterns and should be treated as estimates.
4. Delay prediction is difficult because real-world airline delays depend on live weather, aircraft rotations, crew availability, airport congestion, and other operational factors.
5. The improved delay model performs better than the baseline model, but the R2 remains limited because flight delays are highly unpredictable.
6. The full raw dataset is required to rerun the entire Spark pipeline from scratch.
7. The web app is designed for demonstration and decision support, not commercial flight booking.

Lessons Learned / Problems Faced
--------------------------------
1. Large CSV datasets required Spark-based processing rather than pandas-only processing.
2. Windows Spark setup required careful Java, Hadoop, and winutils configuration.
3. Fare data required quality checks because values and units could vary across files.
4. Delay prediction was challenging because flight delays are affected by many real-time operational factors.
5. Building the web app directly from large raw data was not practical, so the project used precomputed Spark outputs.
6. Loading local JSON directly in the browser caused issues, so Flask was used to serve the dashboard properly.
7. Separating heavy data processing from lightweight web deployment made the final demo faster and more reliable.

Demo Tips
---------
During the presentation, run:

   python app.py

Then open:

   http://127.0.0.1:5000

Recommended demo flow:
1. Start with the dashboard overview.
2. Show top/busiest airport charts.
3. Show delay and fare trend charts.
4. Search a route such as ATL to ORD, JFK to LAX, BWI to ATL, or BWI to LAX.
5. Explain expected delay, estimated fare, carrier comparison, and BUY / CAUTION / AVOID recommendation.
6. Explain that the web app is powered by results.json generated from Spark outputs.
7. Mention that the full raw dataset was approximately 21.2GB, so only sample data and processed outputs are included in the submission package.

Submission Notes
----------------
This submission includes the four required project deliverables:
1. Slides deck describing the work performed.
2. Project source code.
3. Sample data.
4. README.txt file describing how to run the project.

The complete raw dataset is not included because of file size. The included sample data and processed outputs are sufficient to understand the project structure, data format, analysis workflow, and web app behavior.
