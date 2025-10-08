from pandera import Field
from pandera.typing import Series
from blocksnet.utils.validation import DfSchema


class Schema(DfSchema):
    living_area: Series[float] = Field(ge=0)
    population: Series[float] = Field(ge=0, nullable=True)
