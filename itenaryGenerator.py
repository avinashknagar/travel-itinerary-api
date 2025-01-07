import requests
import json
import re
from llama_index.llms.ollama import Ollama
import os
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

class DataFetchingAgent:
    def __init__(self):
        self.api_key = os.getenv('SERPER_API_KEY')
        if not self.api_key:
            raise ValueError("SERPER_API_KEY not found in environment variables")
        self.headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        self.base_url = 'https://google.serper.dev/search'

    def fetch_data(self, query):
        data = json.dumps({"q": query})
        try:
            response = requests.post(self.base_url, headers=self.headers, data=data)
            if response.status_code == 200:
                return response.json()  # Assumes the API returns JSON
            else:
                return {"error": "Failed to fetch data", "status_code": response.status_code}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}


class PreferenceProcessingAgent:
    def __init__(self, preferences):
        """
        Initialize with user preferences
        preferences should be a dictionary containing:
        - budget: float (per day budget)
        - interests: list of strings (e.g., ['museums', 'food', 'outdoor'])
        - trip_length: int (number of days)
        - accessibility_needs: list of strings (optional)
        - preferred_times: dict (optional, e.g., {'start_time': '09:00', 'end_time': '18:00'})
        """
        self.preferences = preferences
        self.default_preferences = {
            'budget': float('inf'),
            'interests': [],
            'accessibility_needs': [],
            'preferred_times': {'start_time': '09:00', 'end_time': '21:00'},
            'start_date': None,
            'end_date': None
        }
        # Merge provided preferences with defaults
        self.preferences = {**self.default_preferences, **self.preferences}
        
        # Calculate trip length from dates
        if self.preferences['start_date'] and self.preferences['end_date']:
            start = datetime.strptime(self.preferences['start_date'], '%Y-%m-%d')
            end = datetime.strptime(self.preferences['end_date'], '%Y-%m-%d')
            self.preferences['trip_length'] = (end - start).days + 1
        else:
            self.preferences['trip_length'] = 1
    
    def process_data(self, data):
        """
        Process and filter the raw API data based on user preferences
        """
        if 'error' in data:
            return data
        
        try:
            processed_results = []
            organic_results = data.get('organic', [])
            
            for result in organic_results:
                # Extract relevant information
                processed_item = {
                    'title': result.get('title', ''),
                    'link': result.get('link', ''),
                    'snippet': result.get('snippet', ''),
                    'position': result.get('position', 0),
                    'relevance_score': 0  # Initial score
                }
                
                # Calculate relevance score based on preferences
                relevance_score = self._calculate_relevance(processed_item)
                processed_item['relevance_score'] = relevance_score
                
                # Only include items that meet minimum relevance threshold
                if relevance_score > 0:
                    processed_results.append(processed_item)
            
            # Sort by relevance score
            processed_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Limit results based on trip length
            daily_items = 3  # Number of activities per day
            max_items = self.preferences['trip_length'] * daily_items
            processed_results = processed_results[:max_items]
            
            return {
                'processed_items': processed_results,
                'total_items': len(processed_results),
                'trip_length': self.preferences['trip_length']
            }
            
        except Exception as e:
            return {
                'error': f'Error processing data: {str(e)}',
                'raw_data': data
            }
    
    def _calculate_relevance(self, item):
        """
        Calculate relevance score for an item based on user preferences
        """
        score = 0
        text_to_analyze = f"{item['title']} {item['snippet']}".lower()
        
        # Check for interest matches
        for interest in self.preferences['interests']:
            if interest.lower() in text_to_analyze:
                score += 2
        
        # Check for accessibility needs
        for need in self.preferences['accessibility_needs']:
            if need.lower() in text_to_analyze:
                score += 1
        
        # Basic budget check (if price information is found in the text)
        if self._check_budget_compatibility(text_to_analyze):
            score += 1
        
        return score
    
    def _check_budget_compatibility(self, text):
        """
        Check if the item appears to be within budget
        This is a simple implementation that could be enhanced with better price extraction
        """
        # Look for common price patterns
        price_indicators = ['free', 'no cost', 'complimentary']
        if any(indicator in text.lower() for indicator in price_indicators):
            return True
            
        # If no price information is found, assume it's compatible
        return True

class ItineraryGenerationAgent:
    def __init__(self):
        llm_provider = os.getenv('LLM_PROVIDER')
        self.time_slots = {
            'morning': '09:00-12:00',
            'afternoon': '12:00-17:00',
            'evening': '17:00-21:00'
        }
        
        try:
            if llm_provider.lower() == 'groq':
                self.llm = self._initialize_groq()
            else:  # default to ollama
                self.llm = self._initialize_ollama()
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}")
            raise e

    def _initialize_ollama(self):
        ollama_api_url = "http://localhost:11434"
        return Ollama(
            model='mistral',
            api_base=ollama_api_url,
            timeout=180,
            request_timeout=180,
            temperature=0.7,
            retry_on_failure=True
        )

    def _initialize_groq(self):
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        return Groq(api_key=groq_api_key)

    def generate_itinerary(self, destination):
        """
        Generate itinerary using selected LLM
        """
        try:
            prompt = f"""
            Create a 3-day travel itinerary for {destination}. Include morning, afternoon, and evening activities.
            Format the response as JSON with this structure:
            {{
                "itinerary": {{
                    "total_days": 3,
                    "daily_schedules": [
                        {{
                            "day": 1,
                            "activities": [
                                {{
                                    "time_slot": "morning",
                                    "activity": {{
                                        "name": "Activity name",
                                        "description": "Brief description",
                                        "duration": "2 hours",
                                        "best_time": "9:00 AM",
                                        "tips": "Quick tip"
                                    }}
                                }}
                            ]
                        }}
                    ]
                }}
            }}
            Keep the response concise and ensure it's valid JSON.
            """
            
            print(f"Generating itinerary for {destination}...")
            
            if isinstance(self.llm, Groq):
                response = self.llm.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a travel itinerary expert."},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                    max_tokens=1024
                )
                response_text = response.choices[0].message.content
            else:  # Ollama
                response = self.llm.complete(prompt)
                response_text = str(response).strip()
            
            # Clean up the response text
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            return json.loads(response_text)
            
        except json.JSONDecodeError as e:
            print("Failed to parse JSON response. Raw response:")
            print(response_text)
            return {
                'error': 'Failed to parse LLM response as JSON',
                'raw_response': response_text
            }
        except Exception as e:
            return {
                'error': f'Failed to generate itinerary: {str(e)}'
            }

class TravelSearchAgent:
    def __init__(self):
        self.api_key = os.getenv('SERPER_API_KEY')
        if not self.api_key:
            raise ValueError("SERPER_API_KEY not found in environment variables")
        self.headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        self.base_url = 'https://google.serper.dev/search'

    def search_flights(self, origin, destination, date_range):
        """
        Search for flights using Serper API
        """
        try:
            query = f"flights from {origin} to {destination} {date_range}"
            data = json.dumps({"q": query})
            
            response = requests.post(self.base_url, headers=self.headers, data=data)
            if response.status_code == 200:
                results = response.json()
                
                # Process and filter flight results
                flights = []
                for result in results.get('organic', []):
                    if any(keyword in result.get('title', '').lower() 
                          for keyword in ['flight', 'airline', 'airways']):
                        flights.append({
                            'title': result.get('title', ''),
                            'link': result.get('link', ''),
                            'snippet': result.get('snippet', ''),
                            'price': self._extract_price(result.get('snippet', ''))
                        })
                
                return {
                    'status': 'success',
                    'flights': flights[:5]  # Return top 5 results
                }
            
            return {'status': 'error', 'message': f"API Error: {response.status_code}"}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def search_hotels(self, destination, check_in, check_out, preferences=None):
        """
        Search for hotels using Serper API
        """
        try:
            # Build query based on preferences
            query = f"hotels in {destination} {check_in} to {check_out}"
            if preferences:
                if preferences.get('budget'):
                    query += f" under {preferences['budget']} per night"
                if preferences.get('amenities'):
                    query += f" with {', '.join(preferences['amenities'])}"
            
            data = json.dumps({"q": query})
            
            response = requests.post(self.base_url, headers=self.headers, data=data)
            if response.status_code == 200:
                results = response.json()
                
                # Process and filter hotel results
                hotels = []
                for result in results.get('organic', []):
                    if any(keyword in result.get('title', '').lower() 
                          for keyword in ['hotel', 'resort', 'inn']):
                        hotels.append({
                            'name': result.get('title', ''),
                            'link': result.get('link', ''),
                            'description': result.get('snippet', ''),
                            'price': self._extract_price(result.get('snippet', '')),
                            'rating': self._extract_rating(result.get('snippet', ''))
                        })
                
                return {
                    'status': 'success',
                    'hotels': hotels[:5]  # Return top 5 results
                }
            
            return {'status': 'error', 'message': f"API Error: {response.status_code}"}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def _extract_price(self, text):
        """
        Extract price from text using regex
        """
        price_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
        matches = re.findall(price_pattern, text)
        return matches[0] if matches else None

    def _extract_rating(self, text):
        """
        Extract rating from text using regex
        """
        rating_pattern = r'(\d\.?\d?)/5'
        matches = re.findall(rating_pattern, text)
        return matches[0] if matches else None

class ItinerarySystem:
    def __init__(self, user_prefs=None):
        print(f"\n[DEBUG] Initializing ItinerarySystem with preferences: {user_prefs}")
        if user_prefs is None:
            print("[DEBUG] No user preferences provided")
            return

        try:
            self.pref_agent = PreferenceProcessingAgent(user_prefs)
            self.itinerary_agent = ItineraryGenerationAgent()
            self.travel_agent = TravelSearchAgent()
            print("[DEBUG] Successfully initialized all agents")
        except Exception as e:
            print(f"[DEBUG] Error initializing agents: {str(e)}")
            raise e

    def create_itinerary(self):
        print("\n[DEBUG] Starting create_itinerary")
        try:
            destination = self.pref_agent.preferences['destination']
            print(f"[DEBUG] Generating itinerary for destination: {destination}")
            
            # Generate itinerary using LLM
            print("[DEBUG] Calling itinerary_agent.generate_itinerary")
            itinerary = self.itinerary_agent.generate_itinerary(destination)
            print(f"[DEBUG] Itinerary generation result: {itinerary.get('error') if 'error' in itinerary else 'Success'}")
            
            if 'error' in itinerary:
                raise Exception(itinerary['error'])
            
            # Search for travel options
            print("\n[DEBUG] Searching for travel options")
            travel_options = self.search_travel_options(
                origin=self.pref_agent.preferences.get('origin', "New York"),
                destination=destination,
                dates=f"{self.pref_agent.preferences.get('start_date')} to {self.pref_agent.preferences.get('end_date')}"
            )
            print(f"[DEBUG] Travel options search result: {travel_options.get('error') if 'error' in travel_options else 'Success'}")
            
            if 'error' in travel_options:
                raise Exception(travel_options['error'])
            
            # Combine itinerary and travel options
            complete_plan = {
                "itinerary": itinerary,
                "travel_options": travel_options
            }
            
            print("[DEBUG] Successfully created complete itinerary")
            return complete_plan

        except Exception as e:
            print(f"[DEBUG] Error in create_itinerary: {str(e)}")
            error_response = {
                'error': str(e),
                'status': 'failed'
            }
            return error_response

    def update_preferences(self, new_preferences):
        """
        Update user preferences after initialization
        """
        self.pref_agent = PreferenceProcessingAgent(new_preferences)

    def search_travel_options(self, origin, destination, dates):
        """
        Search for both flights and hotels
        """
        try:
            print(f"[DEBUG] Searching for flights from {origin} to {destination} for dates {dates}")
            flights = self.travel_agent.search_flights(
                origin=origin,
                destination=destination,
                date_range=dates
            )

            print(f"[DEBUG] Searching for hotels in {destination}")
            hotels = self.travel_agent.search_hotels(
                destination=destination,
                check_in=dates.split(' to ')[0],
                check_out=dates.split(' to ')[1],
                preferences={
                    'budget': self.pref_agent.preferences.get('budget'),
                    'amenities': ['wifi', 'breakfast']  # Example amenities
                }
            )

            return {
                'flights': flights,
                'hotels': hotels
            }

        except Exception as e:
            print(f"[DEBUG] Error in search_travel_options: {str(e)}")
            return {
                'error': str(e),
                'status': 'failed'
            }