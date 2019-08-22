# DeepLearn 2019

Immár harmadik alkalommal képviselem az MTA Nyelvtudományi Intézetét a
DeepLearn nyáriegyetemen, ezúttal Varsóban, azzal az adalékkal, hogy jövő héten
röpülök tovább az ACL-re Firenzébe, szinte közvetlenül -- Amszterdamon át. 
Az első napról már frissiben jegyzetelek, mert könnyen lehet, hogy később más
kiszorítja. 
A DeepLearn szokásaihoz híven öt napon át 5-5 idősávban 2-3 előadás közül lehet
választani, amelyek nyolcszor 2-3 háromalkalmas kurzussá állnak össze.

Az első hármasból nem volt nehéz választanom, mert az egyiket maga Mikolov
fémjelezte.  Nagy izgalommal vártam -- ahhoz hasonlóval, mint a pápát
Csíksomlyón (mármint tényleg, pápista vagyok).

# Mikolov és Bojanowski: ... Modeling and Representing Natural Languages

<div style="text-align: right"> 
                         _In the morning, we developed alorithms like word2vec_
</div>

Mikolov előadása megerősítette, talán egyenesen visszaadta a hagyományos gépi
tanulásba vetett hitemet, ugyanis Tomáš végig az egyszerű, hatékony
baseline-okat hangsúlyozta. Azzal vezette be a témát, hogy a gépi tanulásban
inkrementális a fejlődés, a neurális hálók pedig egy számítási modell,
helytelen például az agyhoz hasonló dologként gondolni rá. 
A fogalmakat kifejezetten régi publikációkhoz kötötte: az n-gram modell kapcsán
Shannon 50-es évek beli munkásságára, a szóvektorok alaptulajdonságainál
pedig Collobert és Westonra (2011, 2007) hivatkozott.  A szóosztályoknak
_(Brown clusters)_ is szánt egy szükségképpen felszínes de talán az újoncnak is
érhető szlájdot, a szózsák modellt pedig azzal ajánlotta, hogy hosszú
dokumentumok esetén ma is a legjobb lehet.  A logisztikus regressziót is
alábecsültnek mutatta be (Joulin how-to-jára hivatkozva), majd a
támvektorgépekre és az őket implementáló szabad szoftverekre (libsvm, svmtorch,
svmlight) hívta fel a figyelmet, mint a logisztikus regresszió valamivel
bonyolultabb, de nem feltétlen jobb változatára.

Az ideghálókat felvezetve kifejtette a biológiai hasonlat elleni megjegyzését:
a valódi neuronok spike-okat (azt hiszem, ezt hívják magyarul akciós
potenciálnak) küldenem egymásnak bizonyos frekvenciákon. 
Az egyrétegű idegháló univerzális számítóereje kapcsán is a hatékonyságra hívta
fel a figyelmet, de ezúttal a bonyolultabb architektúra javára: 
az univerzalitás egy elvi tulajdonság, de egy ponton túl memorizálást jelent a
mélyebb háló általánosítása helyett. A sztochasztikus gradiensleszállásról
(SGD) azt mondta, hogy sok alternatíváját vetették fel, de kellően alapos
kiértékelésben kevés bizonyult jobbnak nála. 
Az ideghálókról szóló általános részt azzal zárta, hogy bizonyos
hiperparamétereket továbbra sem lehet gépi tanulással megválasztani: 
az aktivációs függvényt, a rétegek számát és méretét, a tanulási rátát, a
jellemzőket és a regularizálást. A mély hálók sokszor nem hatékonyak, mert
egyes idegsejtek csak bizonyos tartományokban aktívak (Minsky és Papert 1969).

A szóvektorokról (1.p59) szóló részben, mint egy jó öreg prof, elmondta a
2013-as cikkeket, _they _were_ quite popular_. Az skip-gram hasonló eredményt
ér el, mint a folytonos szózsák (cbow), és gyorsabban teszi.  
Az pontos implementáció körüli kontraverziára, sőt egyáltalán a két
architektúra közötti különbségre, hasonlóan tekint, mint a korábban említett
hiperparaméterekre: ki kell próbálni a különféle lehetőségeket.  Inkább
a korpusz megválasztására irányította a figyelmet. 
A szófordításhoz kapcsolódó későbbi munka: [Joulin+ Loss in Translation:
Learning ... with a Retrieval Criterion](https://arxiv.org/abs/1804.07745)

A második idősávjukban Piotr Bojanowski beszélt a fastText szó alatti
modelljeiről. A motiváló lenyegl példa, ha jól értem egy 'leananászoz' (mármint
ananásznak nevez) jelentésű lengyel szó volt. A fastText ismertetését bőszen
jegyzeteltem, például nem tudtam, hogy a karakter-_n_gramok hashelve vannak a
nagy embeddingmátrix elkerülésére. Újabb fejleményeket is elmondott (Mikolov+
LREC 18, Grave+ 18, Joulin+ fastText.zip 16). Folyt köv.

Utolsó óra: nyelvmodellezés.  Ehhez egyelőre nem fűzök személyes kiemelésket: a
nyelvtechnológusoknak mindenképpen ajánlom a figyelmmébe magukat a szlájdokat.

A Mester külsőre is nagyon más volt, mint amire számítottam: kis termetű,
kicsit lányos.  Korábban azt hallottam róla, hogy több mint zárkózott, a
laptopjába motyogva ad elő. Nekem nem volt bajom az előadói stílusával. 
Csak akkor látszott, hogy nagyon nehéz neki ez a helyzet, amikor a szünetben is
odamentek hozzá kérdezni. Ekkor a padlót nézve hallgatta a kérdezőket, mint aki
gyóntat.  


# Bertrand Thirion (INRIA) Understanding the Brain with Machine Learning

A második hármasban ez a kognitív idegtudományi előadás olyan kutatásokról
szólt, ahol mesterséges ideghálókkal modellezik az agyi látást. Az mesterséges
háló aktivációiból jósolják az fMRI-képet. Ahol a jóslás pontos, ott
feltételezik, hogy az adott agyi terület hasonló számítást végez, mint a
mesterséges háló megfelelő része. Így azt is megállapítják, hogy a különböző
területek a feldolgozás milyen szintjéért felelnek. 
Validálták is a kísérleteiket korábbi módszerekkel, amikből tudni lehet, hogy a
látókéreg mely területei felelnek például a látvány közepéért illetve
pereméért. A mesterséges hálónak a színeket illetve a mozgást feldolgozó részei
feletek meg ezeknek a területeknek. Ez értelmesnek tűnik, hiszen az éles
látásban inkább látjuk a színeket, a szélén pedig mozgásokra figyelünk fel.
http://nilearn.github.io/ Elképzelt képet is próbáltak kiolvasni, aminek a
mentális reprezentációra nézve is van tanulsága: az öt körből álló stimulust,
amik a dobókocka szerinti módon voltak elhelyezve, a kísérleti személy X-alakú
dologként írta le, és a képen látszott is, hogy az elmében X alakban össze
vannak kötve a pöttyök. Ennek az előadásnak a második régészetről eljöttem,
mert nagyon fáradt voltam, kicsit rosszul is voltam, és a szállásra kellett
jönnöm balhézni, de sajnálom, mert olyasmiket ígért, hogy a dekódolás GAN-nal
való javítása (1.p68), a hallás dekódolása (1.p73, Kell+ Neuron 2018), és a
nyelvi terület (a szemantikai tér leképezése, 1.p67).

A harmadik órán olyan modellt mutatott, ami az agyról szóló cikkekben szereplő
szavakat, pl. _olvasás_, kapcsolta össze a cikkhez publikált agyi képalkotásból
jövő adatokkal.  Simításként a szavakat egy előtanított embeddingben való
szomszédaival interpolálták.  Elég komoly gépitanulási/numerikus modelleket
használt, pl.  ICA-t.  Végül felhívta a figyelmet az agyi architektúra és a
mély ideghálók közötti néhány különbségre: az agyban csak 6-7 réteg van, inkább
elosztott mint réteges, és hosszútávú feedback van.


# Aaron Courville: generatív modellek

A kínálat harmadik előadásának azt az előnyét emelném ki, hogy teljes megértés
nélkül is adott intuíciót olyasmikről, mint  az autoregresszív CNN-t (Van der
Oord 16), és a hanghullám hierarchikus szerkezete és a WaveNet tágított
(dilated) konvolúciója közötti összefüggés vagy az eloszlások közelítésén
alapuló módszerek.


# Keynote: [Maria-Florina Balcan (Carnegie Mellon) Inkrementális
klaszterezés](http://papers.nips.cc/paper/8263-data-driven-clustering-via-parameterized-lloyds-families)

Három este keynote volt, ezek közül csak az elsőre mentem be. Keynote-nak nem
neveztem volna, inkább egy jó egyetemi órának a klaszterezésről. 
