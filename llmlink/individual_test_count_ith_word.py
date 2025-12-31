# -*- coding: utf-8 -*-

if __name__ == "__main__":
    text = '''
Tom felt that it was time to wake up; this sort of life might be
romantic enough, in his blighted condition, but it was getting to have
too little sentiment and too much distracting variety about it. So he
thought over various plans for relief, and finally hit upon that of
professing to be fond of Pain-killer. He asked for it so often that he
became a nuisance, and his aunt ended by telling him to help himself and
quit bothering her. If it had been Sid, she would have had no misgivings
to alloy her delight; but since it was Tom, she watched the bottle
clandestinely. She found that the medicine did really diminish, but it
did not occur to her that the boy was mending the health of a crack in
the sitting-room floor with it.
'''
    words = text.split()
    for i, word in enumerate(words):
        print(f"{i}: {word}")