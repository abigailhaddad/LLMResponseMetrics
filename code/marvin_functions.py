from marvin import ai_fn, settings, settings

settings.llm_max_tokens=1500
llm_max_context_tokens=2500
settings.llm_temperature=0.0

@ai_fn
def rate_response(target_answer: str, actual_response: str) -> int:
    """Given `target_answer` and 'actual_response', generates a number on a scale from 1 to 10 comparing their similarity ."""

class ResponseRater:
    def __init__(self, target_answer: str, actual_response: str):
        self.target_answer = target_answer
        self.actual_response = actual_response
        self.rating = rate_response(self.target_answer, self.actual_response)

    def __repr__(self):
        return f"Target Answer: {self.target_answer}\nActual Response: {self.actual_response}\nRating: {self.rating}"


def add_ratings_to_df(df):
    df['response_rater'] = df.apply(lambda row: ResponseRater(row['target_answer'], row['response']), axis=1)
    df['rating'] = df['response_rater'].apply(lambda rater: rater.rating)
    return df
