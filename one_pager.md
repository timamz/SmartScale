# SmartScale

**Business value**: Reduce self-checkout friction for produce by auto-recognizing items from a scale-mounted camera, lowering queue time and increasing pricing accuracy.

**What it does**
- Customer uploads an image (camera simulation) and enters weight.
- System predicts label + confidence + top-K alternatives.
- Price is calculated from a stored price list and saved with the transaction.
- Analytics and monitoring dashboards expose usage and model quality signals.

**Stack**
- FastAPI + Celery + RabbitMQ (async inference)
- TF/Keras VGG16 (Adriana213/vgg16-fruit-classifier)
- PostgreSQL (requests, prices, metrics)
- Streamlit UI
- Grafana dashboards
