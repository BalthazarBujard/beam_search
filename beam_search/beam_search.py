from typing import Any, List, Callable, Dict, Tuple, Optional
import torch
import numpy as np
import copy
from scipy.stats import entropy


#TODO : GIVE POSSIBILITY TO USER TO GIVE SCORING FUNCTION AS PARAM
# the definition of the custom score function by the user has to take as argument a Candidate and use its attributes to compute the score
# TODO : MAKE ATTRIBUTES PRIVATE IN ORDER TO NOT MODIFY THEM BY MISTAKE WITH CUSTOM SCORE
class Candidate():
    def __init__(self,states:List[Any], probs:List[float], terminal_state : Optional[int] = None, score_fn : Optional[Callable] = None):
        self.__states = states #(T,)
        self.__probs = probs #(T,)
        self.terminal_state = terminal_state
        self.terminated = np.isin(states,terminal_state).any() #True if terminal state is assigned to candidate sequence
        self.effective_length = len(self.states) if not self.terminated else np.where(np.array(states)==terminal_state)[0][0]+1 #length of sequence until terminal state is found
        self.score_fn = score_fn
    
    @property
    def states(self):
        return self.__states 
    
    @states.setter
    def states(self, new_states : List[Any]):
        self.__states = new_states
    
    @property
    def probs(self):
        return self.__probs 
    
    @probs.setter
    def probs(self,new_probs : List[float]):
        self.__probs = new_probs
    
    def update(self,state:Any,prob:float):
        #update states and probs
        self.states=self.states+[state]
        self.probs=self.probs+[prob]
        
        #update/check effective length and terminated state
        if not self.terminated :
            self.effective_length = len(self.states)
            if state == self.terminal_state:
                self.terminated = True
        
    
    def compute_prob(self) -> float:
        return np.prod(self.probs[:self.effective_length]) #prod(p(yt|y<t)) until y_t == terminal state
    
    def compute_score(self) -> float:
        probs = self.probs[:self.effective_length]        
        return sum(np.log(probs))/(self.effective_length**0.75) 
    
    @property
    def score(self):
        return self.compute_score() if not self.score_fn else self.score_fn(self)
    
    def __str__(self):
        return f"states : {self.states}\nscore : {self.score}"
    
    
class BeamSearch():
    def __init__(self, transition_fn : Callable, 
                 transition_fn_args : Optional[Dict], 
                 score_fn : Optional[Callable] = None,
                 terminal_state : Optional[int] = None):
        
        self.transition_fn = transition_fn #function to compute probabilities over
        self.transition_fn_args = transition_fn_args #additional arguments for transition function
        self.score_fn = score_fn #User defined score function
        self.terminal_state = terminal_state #equivalent of End Of Sentence token
    
    #returns the nbest sequences amongst beam_width best candidates for each element in the batch
    def __call__(self, x_init : torch.Tensor, beam_width : int, max_len : int, nbest : int = 1) -> List[List[Candidate]]:
        
        assert nbest<=beam_width
        
        B, L0 = x_init.shape #x_init can have an initial state greater than 1 (?)
        
        #init
        #list of (state,score) where state is the sequence of idx and score the total conditional probability = prod p(y_t|y<t)
        candidates =[[Candidate([x_init[b].item()], [1], self.terminal_state, self.score_fn) for _ in range(beam_width)] for b in range(B)] #(B,beam_width)
        
        for _ in range(max_len-1): #-1 because starting state is already included
            # print("-*-*-*-new search step-*-*-*-")
            candidates = self.__search_one_step(candidates)

        best_candidates = [sorted(candidates_group,key=lambda x : x.score, reverse=True)[:nbest] for candidates_group in candidates] #(B,)
        
        return best_candidates
    
    def __search_one_step(self, candidates : List[List[Candidate]]):
        
        beam_width = len(candidates[0])
        
        probs = self.transition_fn(candidates,**self.transition_fn_args) #(B,beam_width,state space size)
        
        
            
        # WE WANT TO MAXIMIZE THE prod(P(Y_t|Y<t)) so before doing find_k_best we need to multiply the prob by the score of the candidate
        #get probabilities (prob of prod(P(y_t-1|y<t-1)))
        
        # TODO : TERMINATED CANDIDATES CAN BE KEPT IF THEIR SCORE IS GREATER THAN OTHER CANDIDATES OTPIONS 
        # BUT WE NEED TO COMPARE THE NEW BEAM_STATES_LIKELIHOOD (OF NON-TERMINATED SEQUENCES) WITH THE TERMINATED ONES
        for batch_index, beams_probs in enumerate(probs) :
            # print("----new batch element--------")
            
            #beams_probs (beam_width, state space size)
            this_candidates = candidates[batch_index]
            
            #if first step all candidates share the same state (i.e. start state) -> dont use  all candidates
            if this_candidates[0].effective_length==1:
                beams_probs = beams_probs[0].unsqueeze(0) #(1, state_space)
                this_candidates=[this_candidates[0]] #(1,)
            
            """ 
            THIS METHOD IS APPARENTLY SLOWER THAN COMPUTING THE BEAM_STATES_LIKELIHOODS BUT ITS EASIER TO MODIFY SCORING FUNCTION
            """
            
            #construct new candidate sequences for every new token possibility
            new_candidates : List[List[Candidate]] = [
                [Candidate(c.states+[new_state],c.probs+[new_prob.item()],self.terminal_state, self.score_fn) for new_state, new_prob in enumerate(beams_probs[idx])] 
                for idx,c in enumerate(this_candidates)
                ] #(beam_width, state_space)
            
            
            scores = torch.tensor([[c.score for c in beam_candidates] for beam_candidates in new_candidates])
            
            best_candidates_idx = self.__find_k_best(scores, beam_width)
            
            best_candidates = [new_candidates[idx[0]][idx[1]] for idx in best_candidates_idx]
            
            candidates[batch_index] = best_candidates
            
            """ 
            END
            """
               
            """ 
            OTHER METHOD COMPUTING LIKELIHOOD WITHOUT CREATING NEW SET OF CANDIDATES
            """
            # candidates_probs = torch.tensor([c.compute_prob() for c in this_candidates], device=beam_probs.device).view(-1,1) #(beam_width,1)
            
            # # scores = torch.log(beam_probs*candidates_probs) #(beam_width, state space)
            # topk_states = self.__find_k_best(scores, beam_width) #(beam_width, 2)
            # # print(topk_states)
            
            # # retrieve state probability from beam_probs
            # topk_probs = beam_probs[topk_states[:,0],topk_states[:,1]].numpy(force=True)
            # #print(topk_probs)
            
            
            # #update candidates with new states
            # updated_candidates = self.__update_candidates(this_candidates, topk_states, topk_probs)
            # #assign updated candidates
            # candidates[batch_index] = updated_candidates
            """ 
            END
            """
            
            
            # for c in updated_candidates:
            #     print(c)
            
                
        
        return candidates
    
    def __update_candidates(self, candidates : List[Candidate], topk_states : np.ndarray, topk_probs : np.ndarray):
        
        #updated_candidates = [[None] for _ in range(len(candidates))]
        candidates_to_update : List[Candidate] = [copy.deepcopy(candidates[idx]) for idx in topk_states[:,0]] #retrieve best candidates to continue
            
        for k,(new_state,prob) in enumerate(zip(topk_states[:,1],topk_probs)):
            candidates_to_update[k].update(new_state, prob)
        
        return candidates_to_update
    
    def __find_k_best(self, probs : torch.Tensor, beam_width : int) -> np.ndarray:
        #probs : (beam_width, state space size)
        
        topk_states_flat = torch.topk(probs.flatten(),beam_width)[1]
        
        #topk_states is an array of indexes corresponding to the best previous states and the new state
        #that way we can continue the candidates that maximizes score and forget non-maximum candidates
        topk_states = np.unravel_index(topk_states_flat.numpy(force=True),probs.shape)
        topk_states = np.column_stack(topk_states) #convert to (beam_width,3) format. 3 is if we work with batched inputs and 2 if unbatched (equal to len(shape))
        
        
        return topk_states
