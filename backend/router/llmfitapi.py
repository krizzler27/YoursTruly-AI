from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from services.llmfit_services import LLMFitServices

router = APIRouter(prefix="/api", tags=["LLMFit"])


@router.get('/models', status_code=status.HTTP_200_OK)
async def models():

    try:
        llmfit = LLMFitServices()
        res = await llmfit.list_installed()
        print(res)
        if len(res)==0:
            return JSONResponse(
                content={"message":"No installed Models, Confirm models are installed with `ollama list` in terminal"}
            )

        return JSONResponse(
            content={"models": res}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"Exception occured": str(e), "type": type(e).__name__}
        )