from enum import unique
from ..indicator_enum import IndicatorEnum
from ..indicator_meta import IndicatorMeta


@unique
class SocialProvisionIndicator(IndicatorEnum):
    # education
    KINDERGARTEN = IndicatorMeta("kindergarten", aggregatable=False)
    SCHOOL = IndicatorMeta("school", aggregatable=False)
    COLLEGE = IndicatorMeta("college", aggregatable=False)
    UNIVERSITY = IndicatorMeta("university", aggregatable=False)
    EXTRACURRICULAR = IndicatorMeta("extracurricular", aggregatable=False)

    # healthcare
    HOSPITAL = IndicatorMeta("hospital", aggregatable=False)
    POLYCLINIC = IndicatorMeta("polyclinic", aggregatable=False)
    AMBULANCE = IndicatorMeta("ambulance", aggregatable=False)
    SANATORIUM = IndicatorMeta("sanatorium", aggregatable=False)
    SPECIAL_MEDICAL = IndicatorMeta("special_medical", aggregatable=False)
    PREVENTIVE_MEDICAL = IndicatorMeta("preventive_medical", aggregatable=False)
    PHARMACY = IndicatorMeta("pharmacy", aggregatable=False)

    # sports
    GYM = IndicatorMeta("gym", aggregatable=False)
    SWIMMING_POOL = IndicatorMeta("swimming_pool", aggregatable=False)
    PITCH = IndicatorMeta("pitch", aggregatable=False)
    STADIUM = IndicatorMeta("stadium", aggregatable=False)

    # social
    ORPHANAGE = IndicatorMeta("orphanage", aggregatable=False)
    NURSING_HOME = IndicatorMeta("nursing_home", aggregatable=False)
    SOCIAL_SERVICE_CENTER = IndicatorMeta("social_service_center", aggregatable=False)

    # service
    POST = IndicatorMeta("post", aggregatable=False)
    BANK = IndicatorMeta("bank", aggregatable=False)
    MULTIFUNCTIONAL_CENTER = IndicatorMeta("multifunctional_center", aggregatable=False)

    # leisure
    LIBRARY = IndicatorMeta("library", aggregatable=False)
    MUSEUM = IndicatorMeta("museum", aggregatable=False)
    THEATRE = IndicatorMeta("theatre", aggregatable=False)
    CULTURAL_CENTER = IndicatorMeta("cultural_center", aggregatable=False)
    CINEMA = IndicatorMeta("cinema", aggregatable=False)
    CONCERT_HALL = IndicatorMeta("concert_hall", aggregatable=False)
    ICE_ARENA = IndicatorMeta("ice_arena", aggregatable=False)
    MALL = IndicatorMeta("mall", aggregatable=False)
    PARK = IndicatorMeta("park", aggregatable=False)
    BEACH = IndicatorMeta("beach", aggregatable=False)
    ECO_TRAIL = IndicatorMeta("eco_trail", aggregatable=False)

    # security
    FIRE_STATION = IndicatorMeta("fire_station", aggregatable=False)
    POLICE = IndicatorMeta("police", aggregatable=False)

    # tourism
    HOTEL = IndicatorMeta("hotel", aggregatable=False)
    HOSTEL = IndicatorMeta("hostel", aggregatable=False)
    TOURIST_BASE = IndicatorMeta("tourist_base", aggregatable=False)
    CATERING = IndicatorMeta("cafe", aggregatable=False)
