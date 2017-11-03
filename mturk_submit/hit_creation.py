%pwd
%cd ~/dev/mctest/mturk_submit
# %%
import urllib
import mturk
m = mturk.MechanicalTurk()

import binascii, hmac
import random
SECRET = b"voxelworld1;p,.r" # Used to approve HITs

def get_token_pair():
    token = '%030x' % random.randrange(16**30)
    token_response = hmac.new(SECRET, binascii.unhexlify(token)).hexdigest()
    return token, token_response

# %%
def create_hit_type(title="Describing virtual scenes",
                    description="You will be presented with virtual environments, and asked to describe what you see.",
                    use_prod=False,
                    reward_usd=None
                    ):
    if reward_usd is None:
        raise ValueError("need to specify a reward")
    elif not isinstance(reward_usd, float):
        raise ValueError("reward must be a float")
    elif reward_usd < 0:
        raise ValueError("reward cannot be negative")
    elif reward_usd > 0.20:
        raise ValueError("reward greater than 20 cents. Is this an accident?")
    elif reward_usd == 0:
        raise ValueError("reward of 0 does not make sense")


    if use_prod:
        queue_url = "https://sqs.us-east-1.amazonaws.com/989786911825/nikita-mturk-prod"
    else:
        queue_url = "https://sqs.us-east-1.amazonaws.com/989786911825/nikita-mturk-sandbox"

    if use_prod:
        r = m.request("RegisterHITType", dict(
            Title = title,
            Description = description,
            Reward={"Amount":reward_usd, "CurrencyCode":"USD"},
            AssignmentDurationInSeconds=2700,
            QualificationRequirement=dict(
                QualificationTypeId="000000000000000000L0",
                Comparator="GreaterThanOrEqualTo",
                IntegerValue=95,
                RequiredToPreview=True,
            ),
        ))
    else:
        r = m.request("RegisterHITType", dict(
            Title = title,
            Description = description,
            Reward={"Amount":reward_usd, "CurrencyCode":"USD"},
            AssignmentDurationInSeconds=2700
        ))

    if r.valid:
        hit_type_id = r['RegisterHITTypeResponse']['RegisterHITTypeResult']['HITTypeId']
    else:
        raise ValueError("didn't register")

    r = m.request("SetHITTypeNotification", dict(
        HITTypeId = hit_type_id,
        Active = True,
        Notification=dict(
            Destination=queue_url,
            Transport="SQS",
            Version="2006-05-05",
            EventType=["AssignmentReturned", "AssignmentAbandoned"],
        ),
    ))

    if not r.valid:
        raise Exception("Couldn't enable sqs for HIT type")

    return hit_type_id

# %%
def create_hit(hit_type_id, use_tomato=False, pool="internal1", max_assignments=1):
    token, token_response = get_token_pair()
    if use_tomato:
        portal_url="https://tomato.banatao.berkeley.edu:9151/portal"
        print(portal_url)
    else:
        portal_url = "https://localhost:8081/portal"
    r = m.request("CreateHIT", dict(
        HITTypeId=hit_type_id,
        LifetimeInSeconds=86400,
        MaxAssignments = max_assignments,
        Question="""
        <ExternalQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd">
            <ExternalURL>{}?pool={}&amp;token={}</ExternalURL>
            <FrameHeight>400</FrameHeight>
        </ExternalQuestion>
        """.format(portal_url, pool, token),
        AssignmentReviewPolicy=dict(
            PolicyName="ScoreMyKnownAnswers/2011-09-01",
            Parameter=[
                {"Key": "AnswerKey",
                 "MapEntry": {"Key": "token_response", "Value": token_response}},
                {"Key": "ApproveIfKnownAnswerScoreIsAtLeast", "Value": 50},
                {"Key": "ApproveReason", "Value": "Our interactive console has recorded that you completed this HIT. It has been auto-approved."},
                {"Key": "RejectIfKnownAnswerScoreIsLessThan", "Value": 50},
                {"Key": "RejectReason", "Value": "Your submission has been flagged for rejection by our automated system. We understand that the system is not perfect, and will manually review all rejections to correct any mistakes. Please contact us if you would like to share any additional information."},
            ]
        )
    ))
    return r

# %%
hit_type_id = create_hit_type(title="Describing virtual scenes", use_prod=True, reward_usd=0.10)
create_hit(hit_type_id, use_tomato=True, pool="external1", max_assignments=8)
