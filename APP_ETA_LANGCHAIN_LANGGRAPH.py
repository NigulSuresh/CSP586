# === IMPORTS ===
import json
import os
import numpy as np
import requests
import pandas as pd
import logging
from flask import Flask, request, jsonify
from datetime import datetime
from sklearn.linear_model import LinearRegression
from langchain.agents import AgentExecutor
from langchain_core.tools import Tool
from langchain_core.tools import StructuredTool
from langchain.agents import create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage 
from typing import Optional
from typing_extensions import Annotated
from pydantic import BaseModel, Field
import joblib
from typing import TypedDict, Union

# === APP SETUP ====
app = Flask(__name__)
app.config["PROPAGATE_EXCEPTIONS"] = True
app.config["DEBUG"] = True

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# === GOOGLE CONFIG ===
GOOGLE_API_KEY = "AIzaSyD_EeB94XohdaNijl_gWqJ2UvR6s7nlQRk"

# === FACTORY PATTERN ===
class ModelFactory:
    @staticmethod
    def get_model():
        if os.path.exists("ml_model.pkl"):
            logger.info("Loading existing model...")
            return joblib.load("ml_model.pkl")
        else:
            logger.info("No model found, training new model...")
            df = fetch_data()
            return train_model(df)

# === STRATEGY PATTERN ===
class HybridETAStrategy:
    def compute_eta(self, m1, m2, g1, g2):
        raise NotImplementedError

class DefaultHybridStrategy(HybridETAStrategy):
    def compute_eta(self, m1, m2, g1, g2):
        return (m1 + m2) * 0.6 + (g1 + g2) * 0.4

class TrafficAwareStrategy(HybridETAStrategy):
    def __init__(self, speed_ratio):
        self.speed_ratio = speed_ratio

    def compute_eta(self, m1, m2, g1, g2):
        # Check if traffic is heavy (speed ratio < 0.8 means slower than normal traffic)
        if self.speed_ratio < 0.8:
            # In heavy traffic, rely more on Google ETA (70%) and less on ML prediction (30%)
            return (m1 + m2) * 0.3 + (g1 + g2) * 0.7
        else:
            # In normal/light traffic, rely more on ML prediction (60%) and less on Google ETA (40%)
            return (m1 + m2) * 0.6 + (g1 + g2) * 0.4

# === COMMAND PATTERN ===
class ETACommand:
    def __init__(self, model):
        self.model = model

    def execute(self, input: Union[str, dict], traffic: dict = None, order_picked_up: bool = False):
        if isinstance(input, dict):
            query = input["input"]
            traffic = input.get("traffic", traffic)
            order_picked_up = input.get("order_picked_up", order_picked_up)
        else:
            query = input
        
        lat1, lon1, lat2, lon2, lat3, lon3 = map(float, query.strip().split(","))
        now = datetime.now()

        if order_picked_up or (lat2 == 0 and lon2 == 0):
            route = get_route_data((lat1, lon1), (lat3, lon3))
            features = pd.DataFrame([[route["distance_km"], now.hour, now.weekday(), route["duration_min"]]],
                                    columns=["distance_km", "hour", "day_of_week", "ors_eta_min"])
            ml_eta = self.model.predict(features)[0]
            hybrid = ml_eta * 0.6 + route["duration_min"] * 0.4
            return {
                "summary": (
                    f"Driver -> Customer: Distance: {route['distance_km']:.2f} km, "
                    f"ORS ETA: {route['duration_min']:.2f} min\n"
                    f"Total Hybrid ETA: {hybrid:.2f} min"
                ),
                "polyline": route["polyline"]
            }
            
        # === Full route with driver -> restaurant -> customer
        r1 = get_route_data((lat1, lon1), (lat2, lon2))
        r2 = get_route_data((lat2, lon2), (lat3, lon3))
        f1 = pd.DataFrame([[r1["distance_km"], now.hour, now.weekday(), r1["duration_min"]]],
                          columns=["distance_km", "hour", "day_of_week", "ors_eta_min"])
        f2 = pd.DataFrame([[r2["distance_km"], now.hour, now.weekday(), r2["duration_min"]]],
                          columns=["distance_km", "hour", "day_of_week", "ors_eta_min"])
        m1 = self.model.predict(f1)[0]
        m2 = self.model.predict(f2)[0]

        # === Choose strategy
        if traffic and "trafficSpeedRatio" in traffic:
            strategy = TrafficAwareStrategy(traffic["trafficSpeedRatio"])
        else:
            strategy = DefaultHybridStrategy()

        hybrid = strategy.compute_eta(m1, m2, r1["duration_min"], r2["duration_min"])

        return {
            "summary": (
                f"Driver -> Restaurant: Distance: {r1['distance_km']:.2f} km, ORS ETA: {r1['duration_min']:.2f} min\n"
                f"Restaurant -> Customer: Distance: {r2['distance_km']:.2f} km, ORS ETA: {r2['duration_min']:.2f} min\n"
                f"Total Hybrid ETA: {hybrid:.2f} min"
            ),
            "polyline": r1["polyline"] + r2["polyline"]
        }


# === DATA & MODEL ===
def fetch_data(limit=1000):
    url = "https://data.cityofchicago.org/resource/wrvz-psew.json"
    data = requests.get(url, params={"$limit": limit}).json()
    df = pd.DataFrame(data)
    df = df.dropna(subset=["trip_start_timestamp", "trip_end_timestamp", "trip_miles"])
    df["trip_start"] = pd.to_datetime(df["trip_start_timestamp"])
    df["trip_end"] = pd.to_datetime(df["trip_end_timestamp"])
    df["trip_duration"] = (df["trip_end"] - df["trip_start"]).dt.total_seconds() / 60
    df["hour"] = df["trip_start"].dt.hour
    df["day_of_week"] = df["trip_start"].dt.dayofweek
    df["distance_km"] = df["trip_miles"].astype(float) * 1.60934
    df = df[(df["trip_duration"] < 45) & (df["distance_km"] > 1.0) & (df["trip_duration"] / df["distance_km"] < 6)]
    df["ors_eta_min"] = df["distance_km"] / 18 * 60
    return df[["distance_km", "hour", "day_of_week", "ors_eta_min", "trip_duration"]].dropna()

def train_model(df):
    X = df[["distance_km", "hour", "day_of_week", "ors_eta_min"]]
    y = df["trip_duration"]
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "ml_model.pkl")
    return model

# === Google UTILITIES ===
def geocode_address(address):
    url = f"https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": GOOGLE_API_KEY}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    if data["status"] != "OK":
        raise ValueError(f"Geocoding failed for {address}: {data['status']}")
    location = data["results"][0]["geometry"]["location"]
    return (location["lat"], location["lng"])

def get_route_data(origin, destination):
    if not all(origin) or not all(destination):
        raise ValueError(f"Invalid coordinates: origin={origin}, destination={destination}")
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{origin[0]},{origin[1]}",
        "destination": f"{destination[0]},{destination[1]}",
        "mode": "driving",
        "key": GOOGLE_API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise ValueError(f"Google Directions API failed: {response.status_code} - {response.text}")
    data = response.json()
    if data["status"] != "OK":
        raise ValueError(f"Directions failed: {data['status']}")

    leg = data["routes"][0]["legs"][0]
    distance_km = leg["distance"]["value"] / 1000
    duration_min = leg["duration"]["value"] / 60
    polyline_encoded = data["routes"][0]["overview_polyline"]["points"]

    # Decode polyline to lat/lng list
    import polyline
    decoded = polyline.decode(polyline_encoded)
    return {
        "distance_km": distance_km,
        "duration_min": duration_min,
        "summary": leg["duration"]["text"],
        "polyline": decoded
    }

# === LANGGRAPH AGENT ===
class ETAState(TypedDict):
    input: str
    traffic: Union[dict, None]
    order_picked_up: bool
    eta_result: Union[str, None]
    polyline: Union[list, None]

def setup_langgraph_agent(model):
    command = ETACommand(model)
    class ETAInput(BaseModel):
        input: Annotated[str, Field(description="Coordinates string in the format 'driver_lat,driver_lng,restaurant_lat,restaurant_lng,customer_lat,customer_lng'")]
        traffic: Optional[dict] = Field(default=None, description="Optional traffic data including trafficSpeedRatio")
        order_picked_up: Optional[bool] = Field(default=False, description="Whether the order is already picked up")
    
    tool = StructuredTool.from_function(
        func=command.execute,
        name="ors_eta_predictor",
        description="Predicts hybrid ETA using ML and Google Maps based on coordinates and traffic.",
        args_schema=ETAInput,
        return_direct=True
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an ETA prediction assistant. Always call the 'ors_eta_predictor' function when given input data."),
        ("human", "Call ors_eta_predictor with the following: input='{input}', traffic={traffic}, order_picked_up={order_picked_up}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    agent = create_openai_functions_agent(llm=llm, prompt=prompt, tools=[tool])
    executor = AgentExecutor(agent=agent, tools=[tool], verbose=False)

    def run(state):
        agent_input = {
            "input": state["input"],
            "traffic": state.get("traffic"),
            "order_picked_up": state.get("order_picked_up", False)
        }
    
        agent_result = executor.invoke(agent_input)
        print("=== Agent raw output ===", agent_result)
    
        # Unwrap StructuredTool result if needed
        if isinstance(agent_result, dict):
            output = agent_result.get("output", {})
            if isinstance(output, dict) and "summary" in output:
                return {
                    "input": state["input"],
                    "eta_result": output["summary"],
                    "polyline": output.get("polyline", [])
                }
    
        return {
            "input": state["input"],
            "eta_result": "No result",
            "polyline": []
        }
    
    

    def should_continue(state: ETAState) -> str:
        return "finish" if state["eta_result"] else "predict_eta"

    builder = StateGraph(ETAState)
    builder.add_node("predict_eta", RunnableLambda(run))
    builder.set_entry_point("predict_eta")
    builder.add_conditional_edges("predict_eta", should_continue, {"finish": END, "predict_eta": "predict_eta"})
    return builder.compile()

# === WEATHER / TRAFFIC ===
def get_weather(lat, lon):
    try:
        response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true")
        if response.status_code == 200:
            data = response.json()
            current = data.get("current_weather", {})
            return {
                "temperature_2m": current.get("temperature", 8.7),
                "wind_speed_10m": current.get("windspeed", 21.1),
                "precipitation": 0,
                "time": current.get("time"),
                "interval": 900
            }
    except Exception as e:
        print("Weather fetch error:", e)
    now = datetime.now()
    return {"time": now.isoformat(timespec='minutes'), "interval": 900, "temperature_2m": 8.7, "wind_speed_10m": 21.1, "precipitation": 0}

def estimate_traffic(hour):
    if 7 <= hour <= 9 or 16 <= hour <= 18:
        return {"currentSpeed": 20, "freeFlowSpeed": 30, "trafficSpeedRatio": 20/30}
    else:
        return {"currentSpeed": 30, "freeFlowSpeed": 30, "trafficSpeedRatio": 1.0}

# === APP START ===
model = ModelFactory.get_model()
graph = setup_langgraph_agent(model)

@app.route("/predict-eta", methods=["GET", "POST"])
def predict_eta():
    try:
        with open("new_order.json", "r") as f:
            input_data = json.load(f)
        
        print("=== Loaded new_order.json ===")
        print("Raw input:", input_data)
        
        if "driver_coords" in input_data:
            driver = input_data["driver_coords"]
        else:
            print("Geocoding driver address:", input_data.get("driver_address"))
            driver = geocode_address(input_data.get("driver_address"))
        
        if "restaurant_coords" in input_data:
            restaurant = input_data["restaurant_coords"]
        else:
            print("Geocoding restaurant address:", input_data.get("restaurant_address"))
            restaurant = geocode_address(input_data.get("restaurant_address"))
        
        if "customer_coords" in input_data:
            customer = input_data["customer_coords"]
        else:
            print("Geocoding customer address:", input_data.get("customer_address"))
            customer = geocode_address(input_data.get("customer_address"))
        
        print("Final coordinates:")
        print("Driver:", driver)
        print("Restaurant:", restaurant)
        print("Customer:", customer)
        print("Formatted query string:", f"{driver[0]},{driver[1]},{restaurant[0]},{restaurant[1]},{customer[0]},{customer[1]}")
        
        query = f"{driver[0]},{driver[1]},{restaurant[0]},{restaurant[1]},{customer[0]},{customer[1]}"
                      
        weather_info = get_weather(driver[0], driver[1])
        traffic_info = estimate_traffic(datetime.now().hour)
        query_input = {
            "input": query,
            "traffic": traffic_info,
            "order_picked_up": input_data.get("order_picked_up", False)
        }
        result = graph.invoke(query_input)

        result["polyline"] = [{"latitude": lat, "longitude": lon} for lat, lon in result["polyline"]]

        return jsonify({
            "driver_address": input_data.get("driver_address") or input_data.get("driver_coords") or "Unknown",
            "restaurant_address": input_data.get("restaurant_address") or input_data.get("restaurant_coords") or "Unknown",
            "customer_address": input_data.get("customer_address") or input_data.get("customer_coords") or "Unknown",
            "order_picked_up": input_data.get("order_picked_up", False),
            "weather": weather_info,
            "traffic": traffic_info,
            "agent_response": result["eta_result"],
            "polyline": result["polyline"]
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)
