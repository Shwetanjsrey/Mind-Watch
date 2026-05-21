from emotion_analyzer import analyze_emotions
import matplotlib.pyplot as plt


def analyze_emotion_trends(posts):
    tracked = ["sadness", "fear", "anger", "joy", "disappointment"]

    trend_data = {emotion: [] for emotion in tracked}

    for post in posts:
        result = analyze_emotions(post)

        for emotion in tracked:
            trend_data[emotion].append(
                result.get(emotion, 0)
            )

    return trend_data


def plot_emotion_trends(posts):
    trends = analyze_emotion_trends(posts)

    x = list(range(1, len(posts) + 1))

    plt.figure(figsize=(10, 6))

    for emotion, values in trends.items():
        plt.plot(x, values, label=emotion)

    plt.xlabel("Post Sequence")
    plt.ylabel("Emotion Score")
    plt.title("MindWatch Emotional Trend Analysis")
    plt.legend()
    plt.grid(True)

    plt.savefig("emotion_trend.png")
    print("Chart saved as emotion_trend.png")