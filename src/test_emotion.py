from emotion_analyzer import analyze_emotions


sample = """
I feel completely drained.
Nothing feels meaningful anymore.
I just want to disappear.
"""

result = analyze_emotions(sample)

print(result)