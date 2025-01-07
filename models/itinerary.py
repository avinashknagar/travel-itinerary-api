from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId

class ItineraryModel:
    def __init__(self):
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client['travel_itinerary']
        self.collection = self.db['itineraries']

    def create_job(self, preferences):
        job = {
            'preferences': preferences,
            'status': 'pending',
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'result': None,
            'error': None
        }
        result = self.collection.insert_one(job)
        return str(result.inserted_id)

    def update_job(self, job_id, status, result=None, error=None):
        update_data = {
            'status': status,
            'updated_at': datetime.utcnow()
        }
        if result:
            update_data['result'] = result
        if error:
            update_data['error'] = error

        self.collection.update_one(
            {'_id': job_id},
            {'$set': update_data}
        )

    def get_job(self, job_id):
        job = self.collection.find_one({'_id': job_id})
        if job:
            job['_id'] = str(job['_id'])
        return job

    def get_all_jobs(self):
        jobs = list(self.collection.find().sort('created_at', -1))
        # Convert ObjectId to string for each job
        for job in jobs:
            job['_id'] = str(job['_id'])
        return jobs 

    def delete_job(self, job_id):
        result = self.collection.delete_one({'_id': job_id})
        return result.deleted_count > 0 