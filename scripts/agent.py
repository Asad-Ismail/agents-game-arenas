import queue
import collections
import threading
import time
from llm_utils import get_llm_action

class ThreadedLLMAgent:
    def __init__(self, model, num_moves=4):
        self.model = model
        self.num_moves = num_moves
        self.request_queue = queue.Queue()
        self.action_queue = collections.deque(maxlen=num_moves*2)  # Store sequence of actions
        self.running = False
        self.thread = None
        
    def start(self):
        """Start the worker thread"""
        self.running = True
        self.thread = threading.Thread(target=self._process_requests)
        self.thread.daemon = True  # Thread will exit when main program exits
        self.thread.start()
        print("LLM worker thread started")
        
    def stop(self):
        """Stop the worker thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        print("LLM worker thread stopped")
    
    def _process_requests(self):
        """Worker thread that processes LLM requests"""
        while self.running:
            try:
                # Get the newest observation from the queue (non-blocking)
                try:
                    # Get the latest observation (clear the queue first)
                    latest_observation = None
                    while not self.request_queue.empty():
                        latest_observation = self.request_queue.get_nowait()
                        self.request_queue.task_done()
                    
                    if latest_observation is not None:
                        # Process the observation with LLM to get multiple actions
                        actions = get_llm_action(latest_observation, model=self.model, num_moves=self.num_moves)
                        # Only update the action queue if we got valid actions back
                        if actions and len(actions) > 0:
                            # Clear the previous action queue and add new actions
                            #self.action_queue.clear()
                            for action in actions:
                                self.action_queue.append(action)
                        
                except queue.Empty:
                    # No observations to process, sleep a bit
                    time.sleep(0.01)
            except Exception as e:
                print(f"Error in LLM worker thread: {e}")
                time.sleep(0.1)
    
    def submit_observation(self, observation):
        """Submit a new observation to be processed asynchronously"""
        # Make a copy of the observation to avoid issues if it changes
        obs_copy = observation.copy()
        # Clear any pending observations - we only want the most recent
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
                self.request_queue.task_done()
            except queue.Empty:
                break  
        # Add the new observation to the queue
        self.request_queue.put(obs_copy)
    
    def get_action(self):
        """Get the next action from the queue, or a default if empty"""
        if not self.action_queue:
            # Return default action if no actions are available
            return ("No-Move", "No-Attack", "Waiting for LLM response")
        # Pop the next action from the left of the queue
        return self.action_queue.popleft()