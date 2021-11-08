# IMPORT STATEMENTS

import numpy # calculations, matrices, etc.
import csv # reading CSV files for data intake
# Bernoulli Naive-Bayes was the text classification algorithm I was introduced
# to doing work with the 20newsgroups dataset, so I'm using it here too.
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, MultinomialNB
# Vectorizing text is a way to use it to train an algorithm. Count vectorizer 
# also allows us to manipulate characters and remove stop words!
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# Classification report and confusion matrix are two ways to display data
from sklearn.metrics import classification_report, plot_confusion_matrix
from matplotlib import pyplot

""" This code reads in the CSV files containing instances of real and fake news
and does the most basic things to organize them. It  cleans the data somewhat, 
removing the date, which isn't important to figuring out if news is real or 
fake. It also adds an indication of which dataset the entry belongs to. """

# Read in real news files using Python's built-in CSV module
realNews = [] # will save the content of the real news
with open('news-sources/real.csv', mode = 'r') as file:
	# Convert to a dict-reader, which is more understandable for the data organized
	csvFile = csv.DictReader(file)
	# Go through the entries, clean them up, and make them correct
	for entry in csvFile:
		entry.pop("date") # get rid of the date: it doesn't inform real/fake
		entry["real"] = 1 # add that the entry is real to help train
		realNews.append(entry) # add entry to the list of real news

# Now, read in fake news files using the same methodology
fakeNews = []
with open('news-sources/fake.csv', mode = 'r') as file:
	csvFile = csv.DictReader(file)
	for entry in csvFile:
		entry.pop("date")
		entry["real"] = 0 # the entry is fake so the 'real' datapoint is 0
		fakeNews.append(entry)

""" This code sets up the vectorizer, which is important for parsing and 
interpreting the data. The parameters used are as follows: max_features caps
the amount of considered words (for time/memory purposes), stop_words removes
commonly-used 'stop-words' that don't contribute to the meaning of the text,
n-gram range considers how many surrounding words impact a word (not exactly
but shorthand). """

# Combine the two datasets together before vectorizing. The order doesn't matter
# because we'll randomize with train_test_split().
dataset = realNews + fakeNews
# Taking out every dictionary item of the same key in a list of dictionaries 
# can be done with iterating over it.
text = [ item['text'] for item in dataset ] # extract the text
labels = [ item['real'] for item in dataset ] # take out real/fake values

vectorizer = CountVectorizer(max_features = 100, stop_words = 'english', ngram_range = (1, 3))
x = vectorizer.fit_transform(text) # vectorize the text in each news article
y = numpy.array(labels) # just an array for 0s and 1s

# Here, we use the train_test_split method to set up the training data.
# test_size designates 25% of our dataset to test the model (75% to train).
# random_state = 30 shuffles the data. It's reproducible thanks to the number.
xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state = 10)

# TRAIN CLASSIFIER(S)

# Training the Bernoulli Naive Bayes classification model
bernoulli = BernoulliNB() 
bernoulli.fit(xTrain, yTrain) # fit the model! train it!
# We can determine the accuracy by testing against test data.
bernoulliAcc = bernoulli.score(xTest, yTest)
print("Bernoulli accuracy: " + str(bernoulliAcc) + "\n") 

""" This code is blocked off because I don't want to delete it, nor do I want
to train the other classifier. It's here if needed. 
# Training the Multinomial Naive Bayes classification model
# This one seems to be a pretty standard model, so I've compared it with the
# Bernoulli model, which uses a different strategy to classify.
multinomial = MultinomialNB()
multinomial.fit(xTrain, yTrain)
multinomialAcc = multinomial.score(xTest, yTest)
print("Multinomial accuracy: " + str(multinomialAcc))
# random_state = 10: Bernoulli 98.36%; Multinomial 91.50%
# random_state = 20: Bernoulli 98.66%; Multinomial 92.34%
# random_state = 30: Bernoulli 98.39%; Multinomial 91.84% """

# BREAKING DOWN OUR DATA: VISUALIZE AND COMPREHEND

# This sorts the prominent features of fake and real news in descending order, 
# which allows us some insight into how sorting was done by the algorithm. 
# argsort() returns an array of indices, not an array.
fakeNewsFlags = bernoulli.feature_log_prob_[0, :].argsort()[::-1]
realNewsFlags = bernoulli.feature_log_prob_[1, :].argsort()[::-1]

print("These fifty words or phrases were most associated with fake news.")
# np.take() is used because argsort() returns indices, not strings.
print(numpy.take(vectorizer.get_feature_names(), fakeNewsFlags[:50]))
print("These fifty words or phrases were most associated with real news.")
print(numpy.take(vectorizer.get_feature_names(), realNewsFlags[:50]))
print("\n")

plot_confusion_matrix(bernoulli, xTest, yTest)
# plot_confusion_matrix(multinomial, xTest, yTest)
pyplot.show()

# TESTING THE ALGORITHM ON REAL-LIFE, MODERN SAMPLES

""" I've collected some articles from the internet that are real and fake. These 
aren't from the same time period as the dataset, making the algorithm
more likely to have errors, especially when modern issues come into play. """

# The following three articles are real, taken from October/November 2021
testStringOneReal = "(Reuters) - Minneapolis Mayor Jacob Frey won a second term in Tuesday's election, emerging from a crowded field of candidates after a tumultuous year dominated by the aftermath of George Floyd's murder by a white city police officer, city election results showed on Wednesday. Frey, 40, a Democrat, had opposed a ballot measure – backed by his more liberal rivals – that would replace the police department with a new public safety agency. Instead, he charted a middle ground, calling for police reforms while also vowing to hire more officers for a department that has been hit hard by departures and has struggled to curb a spike in violent crime. Minneapolis employs a ranked-choice system for its mayoral election, in which voters can rank up to three candidates in order of preference. Frey led after the first round of votes were counted on Tuesday, but he fell short of the mark needed for an outright victory. He cemented his win in a subsequent round of counting on Wednesday."
testStringTwoReal = "Nov 4 (Reuters) - A federal judge on Thursday rejected a lawsuit by Jeff Bezos' space company Blue Origin against the U.S. government over NASA's decision to award a $2.9 billion lunar lander contract to rival billionaire Elon Musk's SpaceX. Judge Richard Hertling of the U.S. Court of Federal Claims in Washington granted the government's motion to dismiss the suit filed on Aug. 16. The judge's opinion explaining his reasoning was sealed, as were many other documents in the case, pending a meeting this month on proposed redactions. Blue Origin, created by Amazon.com Inc founder Bezos, expressed disappointment. \”Not the decision we wanted, but we respect the court’s judgment, and wish full success for NASA and SpaceX on the contract,\” Bezos wrote on Twitter. NASA said on Thursday \“it will resume work with SpaceX\” on the lunar lander contract \”as soon as possible.\” The space agency added it \”continues working with multiple American companies to bolster competition and commercial readiness for crewed transportation to the lunar surface.\” NASA halted work on the lunar lander contract through Nov. 1, part of an agreement among the parties to expedite the litigation schedule, which culminated in Thursday's ruling. The U.S. Government Accountability Office in July sided with the NASA over its decision to pick a single lunar lander provider, rejecting Blue Origin's protest. SpaceX, headed by Tesla Inc. Chief Executive Musk, joined the proceedings as an intervener shortly after the lawsuit was filed. NASA had sought proposals for a spacecraft that would carry astronauts to the lunar surface under its Artemis program to return humans to the moon for the first time since 1972. NASA said on Thursday \”there will be forthcoming opportunities for companies to partner with NASA in establishing a long-term human presence at the Moon under the agency’s Artemis program, including a call in 2022 to U.S. industry for recurring crewed lunar landing services.\” SpaceX did not immediately comment."
# This article interestingly registers as a fake news even though it's real.
testStringThreeReal = "WASHINGTON (AP) — Republicans plan to forcefully oppose race and diversity curricula — tapping into a surge of parental frustration about public schools — as a core piece of their strategy in the 2022 midterm elections, a coordinated effort to supercharge a message that mobilized right-leaning voters in Virginia this week and which Democrats dismiss as race-baiting. Coming out of Tuesday’s elections, in which Republican Glenn Youngkin won the governor’s office after aligning with conservative parent groups, the GOP signaled that it saw the fight over teaching about racism as a political winner. Indiana Rep. Jim Banks, chairman of the conservative House Study Committee, issued a memo suggesting \“Republicans can and must become the party of parents.\” House Minority Leader Kevin McCarthy announced support for a \“Parents’ Bill of Rights\” opposing the teaching of \“critical race theory,\” an academic framework about systemic racism that has become a catch-all phrase for teaching about race in U.S. history. \“Parents are angry at what they view as inappropriate social engineering in schools and an unresponsive bureaucracy,\” said Phil Cox, a former executive director of the Republican Governors Association. Democrats were wrestling with how to counter that message. Some dismissed it, saying it won’t have much appeal beyond the GOP’s most conservative base. Others argued the party ignores the power of cultural and racially divisive debates at its peril. They pointed to Republicans’ use of the \“defund the police\” slogan to hammer Democrats and try to alarm white, suburban voters after the demonstrations against police brutality and racism that began in Minneapolis after the killing of George Floyd. Some Democrats blame the phrase, an idea few in the party actually supported, for contributing to losses in House races last year. If the party can’t find an effective response, it could lose its narrow majorities in both congressional chambers next November. The debate comes as the racial justice movement that surged in 2020 was reckoning with losses — a defeated ballot question on remaking policing in Minneapolis, and a series of local elections where voters turned away from candidates who were most vocal about battling institutional racism. \“This happened because of a backlash against what happened last year,\” said Bernice King, the daughter of the the late civil rights leader Rev. Martin Luther King Jr. who runs Atlanta’s King Center. King warned attempts to roll back social justice advances are “not something that we should sleep on.” \“We have to be constantly vigilant, constantly aware,\” she said, \“and collectively apply the necessary pressure where it needs to be applied to ensure that this nation continues to progress.\” Banks’ memo included a series of recommendations on how Republicans aim to mobilize parents next year, and many touch openly on race. He proposed banning federal funding supporting critical race theory and emphasizing legislation ensuring schools are spending money on gifted and talented and advanced placement programs \“instead of exploding Diversity, Equity and Inclusion administrators.\” The coming fight in Congress over the issue was previewed last month, when Attorney General Merrick Garland appeared before two committees to defend a Justice Department directive aimed at protecting school officials who faced threats amid the heated debate over teaching race. Republicans accused Garland accused of targeting conservative parents. Democrats plan to combat such efforts by noting that many top Republicans’ underlying goal is removing government funding from public schools and giving it to private and religious alternatives. They also see the school culture war squabbles as likely to alienate most voters since the vast majority of the nation’s children attend public schools. \“I think Republicans can, will continue to try to divide us and don’t have an answer for real questions about education,\” said Marshall Cohen, the Democratic Governors Association’s political director. \“Like their plan to cut public school funding and give it to private schools.\” White House deputy press secretary Karine Jean-Pierre accused Republicans of \“cynically trying to use our kids as a political football.\” But Jean-Pierre also took on conservatives’ critique that critical race theory teaches white children to be ashamed of their country. \“Great countries are honest, right? They have to be honest with themselves about the history, which is good and the bad,\” she told reporters. \“And our kids should be proud to be Americans after learning that history.\” Most schools don’t teach critical race theory, which centers on the idea that racism is systemic in the nation’s institutions and that they function to maintain the dominance of white people. But parents organizing across the country say they see plenty of examples of how schools are overhauling the way they teach history and gender issues — which some equate with deeper social changes they do not support. And concerns over what students are being taught — especially after remote learning amid the coronavirus pandemic exposed a larger swath of parents to curricula — led to other objections about actions taken by schools and school boards. Those including COVID safety protocols and policies regarding transgender students. \“I’m sure that most people have no problem with teaching history in a balanced way,\” said Georgia Democratic Rep. Hank Johnson. \“But when you say critical race theory, and you say that it is attacking us and causing our children to feel bad about themselves, that is an appeal that is attractive. And, unfortunately for Democrats, it’s hard to defend when someone accuses you of that.\” Democrats were wiped out Tuesday in lower-profile races in Bucks County, Pennsylvania, where critical race theory was a dominant issue at contentious school board meetings for much of the summer and fall. Patrice Tisdale, a Jamaican-born candidate for magisterial district judge, said she felt the political climate was racially charged. She heard “dog whistles” from voters, who called her “antifa” and accused her of wanting to defund the police, she said. While canvassing a neighborhood in the election’s closing weeks, one voter asked Tisdale whether she believed in critical race theory. \“I said, ’What does that have to do with my election?’\” recalled Tisdale, an attorney, who lost her race. \“I’m there all by myself running to be a judge and that was her question.\” The issue had weight in Virginia, too. A majority of voters there — 7 in 10 — said racism is a serious problem in U.S. society, according to AP VoteCast, a survey of Tuesday’s electorate. But 44% of voters said public schools focus \“too much\” on racism in the U.S., while 30% said they focus on racism \“too little.\” The divide along party lines was stark: 78% of Youngkin voters considered the focus on racism in schools to be too much, while 55% of voters for his opponent, Democrat Terry McAuliffe, said it was too little. Youngkin strategist Jeff Roe described the campaign’s message on education as a broad, umbrella issue that allowed the candidate to speak to different groups of voters — some worried about critical race theory, others about eliminating accelerated math classes, school safety and school choice. \“It was about parental knowledge,\” he said. McAuliffe went on the attack last week, portraying Republicans as wanting to ban books. He accused Youngkin of trying to \“silence\” Black authors during a flareup over whether the themes in Nobel laureate’s Toni Morrison’s 1987 novel \“Beloved\” were too explicit. McAuliffe still lost a governor’s race in a state President Joe Biden carried easily just last year. Republican Minnesota Rep. Tom Emmer bristled at equating a movement to defend school \“parental rights\” and race. \“The way this was handled in Virginia was frankly about parents, mothers and fathers, saying we want a say in our child’s education,\” said Emmer, chairman of the National Republican Congressional Committee. That didn’t rattle some Democrats, who see the GOP argument as manufactured and fleeting. \“Republicans are very good at creating issues,\” deadpanned Democratic Michigan Sen. Debbie Stabenow. \“We’ll have to address it, and then they’ll make up something else.\” Beaumont reported from Des Moines, Iowa; Morrison from New York. Associated Press writers Steve Peoples in Doylestown, Pennsylvania, Jill Colvin in New York and Kevin Freking, Mary Clare Jalonick and Hannah Fingerhut in Washington contributed to this report."

# These are fake examples found online. They aren't as recent. 
testStringOneFake = "Democratic freshman Rep. Ilhan Omar (D., Minn.) has been holding a series of secret fundraisers with groups that have been tied to the support of terrorism, appearances that have been closed to the press and hidden from public view. The content of these speeches, given to predominately Muslim audiences, remains unknown, prompting some of Omar's critics to express concern about the type of rhetoric she is using before these paying audiences, particularly in light of the lawmaker's repeated use of anti-Semitic tropes in public. Omar recently spoke in Florida at a private event hosted by Islamic Relief, a charity organization long said to have deep ties to groups that advocate terrorism against Israel. Over the weekend, she will appear at another private event in California that is hosted by CAIR-CA PAC, a political action committee affiliated with the Council on American Islamic Relations, or CAIR a group that was named as an unindicted co-conspirator in a massive terror-funding incident. Omar's appearance at these closed-door forums is raising eyebrows in the pro-Israel world due to her repeated and unapologetic public use of anti-Semitic stereotypes accusing Jewish people of controlling foreign policy and politics. With Omar's popularity skyrocketing on the anti-Israel left, it appears her rhetoric is translating into fundraising prowess. It remains unclear what Omar has told these audiences in her private talks. Washington Free Beacon attempts to obtain video of past events were unsuccessful, and multiple local news and television outlets in the Tampa, Fla., area, where Omar spoke to Islamic Relief last month, confirmed they were unable to gain access to the closed door event. Islamic Relief has come under congressional investigation for what lawmakers have described as its efforts to provide assistance to terrorist group such as Hamas and the Muslim Brotherhood. The charity has been banned by some countries as a result of these ties. In 2017, Congress sought to ban taxpayer funds from reaching the charity due to these terror links. A representative from Islamic Relief declined to provide the Free Beacon with any material related to Omar's appearance. \”The event was closed to the media. No materials are available,\” the official said. On Sunday, Omar will hold another meet and greet in Irvine, Calif., for CAIR-CA PAC. Those wishing to hear Omar speak are being asked to donate anywhere from $50 to $250 dollars, according to a flyer for the event. The CAIR event also appears closed to the press. Free Beacon attempts to contact the organizer and obtain access were unsuccessful. Requests for comment on the nature of the speeches sent to Omar's congressional office also were not returned. CAIR, a Muslim advocacy group known for its anti-Israel positions, was named by the U.S. government as part of a large network of groups known to be supporting Hamas. CAIR has been cited by the Anti-Defamation League, or ADL, for using its network of supporters to promote an \”anti-Israel agenda.\” \”CAIR’s anti-Israel agenda dates back to its founding by leaders of the Islamic Association for Palestine (IAP), a Hamas affiliated anti-Semitic propaganda organization,\” according to the ADL. \”While CAIR has denounced specific acts of terrorism in the U.S. and abroad, for many years it refused to unequivocally condemn Palestinian terror organizations and Hezbollah by name, which the U.S. and international community have condemned and isolated.\” Sarah Stern, founder of president of the Endowment for Middle East Truth, or EMET, a pro-Israel group that has condemned Omar for promoting anti-Semitic conspiracy theories, told the Free Beacon that the private nature of these events before controversial Islamic groups is very concerning. \”I just wonder what is Rep. Omar saying to a group of Islamic supporters that she feels is so secretive that she cannot say it to the American people, as a whole?\” Stern wondered. \“What is so secretive that it has to be off the record and closed to reporters? Will she say the same things in public to her Jewish voters in Minnesota that she says to her Islamic friends? What does this tell us about her openness, her honesty and her integrity?\” One veteran Republican political operative expressed concern about the secretive nature of these talks, telling the Free Beacon that Democrats must hide behind-closed-doors to appease these groups with anti-Israel rhetoric. \”Of course she's holding these meetings in secret. That's just how Democrats roll these days,\” the source said. \”They're for limiting your ability to spend money on the candidates you want to support, and for secretly fundraising from Islamist groups who support them. It really puts their support for campaign finance reform into perspective.\” After last month's Islamic Relief event, Stern's EMET and many other pro-Israel groups penned a letter to Democratic leaders in the House demanding Omar's removal from the powerful House Foreign Affairs Committee. These groups argued that Omar's anti-Semitic rhetoric and secretive meetings should disqualify her from a seat on that committee, which oversees the U.S.-Israel military alliance. \”Rep. Omar's presence as a keynote speaker to raise funds for Islamic Relief USA, whose parent organization and chapters have documented ties to terrorist organizations, demonstrates that she has learned next to nothing over the last few weeks when she was reprimanded by your office and by other Democrats for posting ugly, anti-Semitic attacks on Jews and their organizations,\” the pro-Israel groups wrote in a letter send to House Foreign Affairs Committee chair Elliott Engel (D., N.Y.) and House Speaker Nancy Pelosi (D., Calif.)."
testStringTwoFake = "Junior Congresswoman Alexandra Ocasio-Cortez has made a lot of waves recently with her Green New Deal nonsense and child-like proposals and speeches.  Well, it’s no different this afternoon as the political pixie announced her intention to draft legislation banning motorcycles from use in the United States of America. Both Clay and Jax Teller take time off from their busy schedule of runnin’ guns, lovin’ women, and threatening Henry Rollins to address the issue. The Senorita of Socialism threw out all manner of statistics regarding deadly accidents and injuries, relaxed traffic rules and tolls for bikers, as well as a not-so-veiled jab at a core demographic of President Trump’s supporters: \“Besides like, what I just said?  A lot of these like, motorcycle people, okay, they’re like : ‘Ooh, look at me, I’m all old and fat and tough and I voted for Trump and smell like wet dog.’  And I’m supposed to slow my Prius down so you can like, noise pollute past everyone?  I mean some of us have a nail appointment, people.\” Opposing the ban, spokesman and leader of \“Bikers For Trump\”, Clee Torres, also gave a statement to the press, expressing his, and his organization’s views : \“I don’t know where they found this little girl with the cute little mouth, but ain’t nobody taking away our hogs.  President Trump is the best damn thing to happen to this country since Zima.\” Torres added that Bikers For Trump would hold a protest against the proposal with a million motorcycle convoy in Washington, previous incarnations of which have drawn tens of participants.  Cortez may want to get herself back behind the bar where her little Punky Brewster act can make her some tips."
testStringThreeFake = "Newly elected Congresswoman Ilhan Omar has proposed a nationwide ban on something…unheard of. Bacon. The pork product bagel sandwiches are built on. The staple of the American soldier, and the most recognizable smell at Shoney’s breakfast buffet. Ilhan Omar wants to take your bacon away. She says it’s for our own good; that our health depends on it. She also says the way we keep and slaughter pigs is immoral and needs to be addressed. Unfortunately, we know that this is just another case of Musslamic Shakira Law in action. The Quaran says: \“Only the American Infidel Eats Bacon, and he Shall Pay the Ultimate Price Before Allahu.\” That sounds like a threat, doesn’t it? It says that. I swear. Omar’s people put out a statement that walked back her attack on bacon a bit: \“No…you idiots. We should consider rethinking how much we eat for our own health and ‘yous wants to take mah baconz’ are two completely different things. You heard it wrong, you then turned it into bullhonkey propaganda. I said bacon isn’t good for you. I don’t eat it. I also said, three days earlier, that slaughterhouses mistreat animals. They do. I was told there would be a lot of twisting and ignorance but I have to say…the level is a bit upsetting.\” Oh, ok sure. Now it’s all our fault because of things you said. Try to remember, Omar — a person can always be judged by what they say. Mitch McConnell says Omar’s ridiculous bill to tear bacon from the arms of Americans — especially our soldiers who rely on it for a taste of home — was dead before she even presented it. Omar reminded him that there was no bill, just an opinion. McConnell said her insolence may be cause for impeachment."

# This method lets the algorithm test a specific piece of news, which will be
# done with all of the above samples.
def test(articleText):
	""" Test an article using the Bernoulli classifier. """
	# print(articleText)
	aT = vectorizer.transform([articleText]) # vectorizer needs something iterable
	if bernoulli.predict(aT)[0] == 1: # it outputs [0] or [1] for fake or real
		print("I'm guessing that this article is real.")
	else:
		print("I'm guessing that this article is fake.")

# Test the samples
print("These articles are real.")
test(testStringOneReal)
test(testStringTwoReal)
test(testStringThreeReal) # our false negative
print("\nThese articles are fake.")
test(testStringOneFake)
test(testStringTwoFake)
test(testStringThreeFake)

# DONE :)