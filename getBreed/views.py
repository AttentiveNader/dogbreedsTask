from django.shortcuts import render
from rest_framework.response import Response

from django.http import JsonResponse
from PIL import Image
import json
import io
import base64
from django.views.decorators.csrf import csrf_exempt
from .apps import GetbreedConfig

# Create your views here.


def index(request):
    return render(request,"base.html")

@csrf_exempt
def getBreed(request):
    # print(request.body)
    body_unicode = request.body.decode('utf-8')
    message = json.loads(body_unicode)
    # print(message)

    # message = request.get_json(force=True)
    # image =
    im_bytes = base64.b64decode(message["image"])
    # buffer = base64.decodebytes(image)
    # image /= np.frombuffer(buffer, dtype=np.float64)
    im_file = io.BytesIO(im_bytes)

    # print(image)
    score,label = GetbreedConfig.predictor.predict(Image.open(im_file))

    return JsonResponse({"breed":label,"score":score})



