// Sounds the model knows. Built from the 200 FSD50K class labels the U-Net was
// trained on (FSD50K.ground_truth/vocabulary.csv), with short names (<= 15
// chars, the prompt limit) and broad parent classes like "Animal" or "Music"
// left out. After the "|" come other words people might type for the same
// sound; those autocorrect to the name.
const RAW = `
engine revving|revving,rev,vroom,accelerating,accelerate
accordion
acoustic guitar|acoustic
aircraft|helicopter,chopper,rocket,spaceship,plane,airplane,aeroplane,jet,fixed wing aircraft
alarm|alarm clock,beeping
applause|applauding,clap clap
bark|barking,woof,woofing
bass drum|kick drum,kick
bass guitar|bass
bathtub|bath,tub
bell|bells,ding,ring
bicycle|bike,cycling
bicycle bell|bike bell
bird|birds,birdsong,bird song,bird call,birdcall
boat|boats,ship,water vehicle
boiling|boil,kettle,lava,cauldron,potion
boom|booming
violin|cello,fiddle,strings,viola,bowed string
brass|horn,tuba,trombone,brass instrument
breathing|breath,breathe,panting,pant
burp|burping,belch
bus|buses
buzz|buzzing,bzz,laser,zap,electricity
camera|shutter,camera shutter
car|cars,automobile
car passing by|passing car,car passing
cat|cats,kitten,kitty
chatter|chattering,babble,talking
cheering|cheer,cheers,hooray
chewing|chew,eating,munching,crunching
chicken|rooster,hen,cluck,clucking,cock
child speaking|kid,kids,child,children,baby
chime|chimes,chiming
clink|clinking,chink,tink,toast
tweet|chirp,chirping,tweeting
chuckle|chortle,snicker
church bell|church bells,cathedral
clapping|clap,claps
clock|clocks
coin|coins,dropping coin,money
keyboard typing|computer,computer keyboard,laptop
conversation|talk,chat,discussion
cough|coughing
cowbell
crack|cracking
crackle|crackling
crash cymbal|crash
cricket|crickets
crow|crows,raven,caw
crowd|crowds,audience
crumpling|crinkle,crinkling,crumple,paper
crushing|crush,crunch
crying|cry,sobbing,sob,weeping
cupboard|cabinet,closet
cutlery|silverware,fork,knife,spoon
cymbal|cymbals
dishes|pots,pans,pots and pans,dish
dog|dogs,puppy,pup,doggy,hound
door|doors,door slam
doorbell|door bell
drawer|drawers
drill|drilling
drip|dripping,drips
drum|drums,drumming
drum kit|drumkit,drum set
electric guitar|guitar solo
engine|engines,motor,motors
engine starting|ignition,starting car
explosion|volcano,dynamite,grenade,explode,exploding,blast,bomb,kaboom
fart|farting,toot
female singing|woman singing,singer,soprano
woman speaking|female speech,woman,lady,female
fill|filling,filling water
finger snapping|snap,snapping,snaps
fire|campfire,flames,flame,bonfire
fireworks|firework
frog|frogs,ribbit,toad,croak
frying|fry,sizzle,sizzling,cooking
gasp|gasping
giggle|giggling
glass|glasses
glockenspiel
gong
growl|growling,snarl,dragon,monster,lion,tiger,bear,wolf,zombie
guitar|guitars
seagull|gull,gulls,seagulls
gunshot|gunfire,gun,guns,shooting,shot,pistol,rifle
gurgling|gurgle
hammer|hammering
hands
harmonica
harp
hi-hat|hihat,hi hat
hiss|hissing,snake
human voice|voice,voices,hum,humming
idling|idle,idling engine
insect|insects,bug,bugs,fly,bee,bees,mosquito
synth|synthesizer,musical keyboard,electric piano
keys jangling|keys,jangle,keychain
knock|knocking,knocks
laughter|laugh,laughing,lol,haha
liquid|fluid
male singing|man singing,tenor,baritone
man speaking|male speech,man,guy,male
mallet|mallets,mallet percussion
xylophone|marimba,vibraphone
fan|ceiling fan,mechanical fan
meow|meowing,miaow,mew
microwave|microwave oven
motorcycle|motorbike,moped,scooter
music|song,melody,tune
ocean|sea,beach
organ|pipe organ
tape|duct tape,packing tape
percussion
piano|keyboard,grand piano
pluck|plucking,plucked string
pour|pouring
power tool|power tools
printer|printing
purr|purring
race car|racecar,racing,formula one,f1
train|trains,railway,rail,locomotive
rain|raining,rainy,rainfall,drizzle,downpour
raindrop|raindrops,rain drop
ratchet|wrench
rattle|rattling,maraca,maracas,shaker
ringtone|ring tone
run|running,jogging,sprint
sawing|saw,chainsaw
scissors|cutting,snip
scratching|scratch,dj scratch,turntable
screaming|scream,screams,shriek
screech|screeching,squeal,brakes
shatter|shattering,breaking glass,smash
shout|shouting
sigh|sighing
singing|sing,choir,vocals
sink|washing up
siren|sirens,ambulance,police,police siren,fire truck
skateboard|skate,skating,skateboarding
slam|slamming
sliding door|automatic door
snare drum|snare
sneeze|sneezing,achoo
speech|speaking,speak,talking voice,narration
robot voice|speech synthesizer,robot,text to speech,tts
splash|splashing,splatter
squeak|squeaking,squeaky,mouse
stream|creek,brook,river
strum|strumming
subway|metro,underground,tube
tabla
tambourine
tap|tapping
tearing|tear,ripping,rip
telephone|phone,phone ringing,landline,cellphone
thud|thump,thumping
thunder|thunderclap,lightning
thunderstorm|storm,storms
tick|ticking
tick-tock|tick tock,ticktock
toilet flush|toilet,flush,flushing
tools|toolbox,workshop
traffic|road,highway,roadway,street
trickle|dribble,trickling
truck|trucks,lorry
trumpet|trumpets,bugle
typewriter|typewriting
typing|type,keyboard keys
car horn|honk,honking,horn honk,vehicle horn
footsteps|walking,walk,steps,footstep
water|h2o
faucet|tap water,water tap
waves|wave,surf,surfing
whisper|whispering,whispers,asmr
whoosh|swoosh,swish,woosh,swoop,ninja,sword,karate,swing
wind|windy,breeze,gust,ghost,spooky,haunted
wind chime|wind chimes,windchime
flute|woodwind,clarinet,oboe,saxophone,sax,recorder
wood|wooden,log,logs
writing|pen,pencil,scribble
yell|yelling
zipper|zip,unzip
`

export interface Sound {
  name: string
  aliases: string[]
}

export const SOUNDS: Sound[] = RAW.trim()
  .split('\n')
  .map((line) => {
    const [name, rest] = line.split('|')
    return { name, aliases: rest ? rest.split(',') : [] }
  })
