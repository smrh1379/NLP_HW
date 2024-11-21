from opsdroid.skill import Skill
from opsdroid.matchers import match_event
from opsdroid.events import UserInvite, JoinRoom,Message

class AcceptInvites(Skill):
    @match_event(UserInvite)
    async def user_invite(self, invite):
        print("\n--USER Invite---\n")
        print(f"user invite -> {invite}")
        if isinstance(invite, UserInvite):
            await invite.respond(JoinRoom())
            await invite.respond(Message("Thank you for inviting me! How can I assist you? 😊"))