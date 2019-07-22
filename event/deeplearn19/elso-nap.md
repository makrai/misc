Immár harmadik alkalommal képviselem az MTA Nyelvtudományi Intézetét a
DeepLearn nyáriegyetemen, ezúttal Varsóban, azzal az adalékkal, hogy jövő héten
röpülök tovább az ACL-re Firenzébe, szinte közvetlenül -- Amszterdamon át. Az
első napról már frissiben jegyzetelek, mert könnyen lehet, hogy később más
kiszorítja. A DeepLearn szokásaihoz híven öt napon át 5-5 idősávban 
2-3 előadás közül lehet választani, amelyek nyolc háromalkalmas kurzussá állnak
össze.

Az első hármasból nem volt nehéz választanom, mert az egyiket maga Mikolov
fémjelezte.  Nagy izgalommal vártam -- ahhoz hasonlóval, mint a pápát
Csíksomlyón (mármint tényleg, pápista vagyok).

Mikolov előadása megerősítette, talán egyenesen visszaadta a hagyományos gépi
tanulásba vetett hitemet, ugyanis Tomáš végig az egyszerű, hatékony
baseline-okat hangsúlyozta. Azzal vezette be a témát, hogy a gépi tanulásban
inkrementális a fejlődés (a neurális hálók pedig egy számítási modell,
helytelen az agyhoz hasonló dologként gondolni rá). A fogalmakat kifejezetten
régi publikációkhoz kötötte: az n-gram modell kapcsán Shannonra 50-es évek beli
munkásságára, a szóvektorok alaptulajdonságainál pedig Collobert és Westonra
(11, 07) hivatkozott.  A szóosztályoknak is szánt egy szükségképpen felszínes
de talán az újoncnak is érhető szlájdot, a one-hot ábrázolást pedig azzal
ajánlotta, hogy hosszú dokumentumok esetén ma is a legjobb lehet. A logisztikus
regressziót is alábecsültnek mutatta be (Joulin how-to-jára hivatkozva), majd a
tartóvektorgépekre és az őket implementáló szabad szoftverekre (libsvm,
svmtorch, svmlight) hívta fel a figyelmet, mint a logisztikus regresszió
valamivel bonyolultabb, de nem feltétlen jobb változatára.

Az ideghálókra felvezetve pontosított a biológiai hasonlat elleni megjegyzését:
a valódi neuronok spike-okat (azt hiszem, ezt hívják magyarul akciós
potenciálnak) küldenem egymásnak bizonyos frekvenciákon. Az egyrétegű idegháló
univerzális számítóereje kapcsán is a hatékonyságra hívta fel a figyelmet, de
ezúttal egy egyszerű architektúra veszélyével: az univerzalitás egy elvi
tulajdonság, de egy ponton túl memorizálást jelent a mélyebb háló
általánosítása helyett. A sztochasztikus gradiensleszállásról (SGD) azt mondta,
hogy sok alternatívát felvetettek, de kevés esetben bizonyult jobbnak az
SGD-nél kellően alapos kiértékelésben. Az ideghálókról szóló általános részt
azzal zárta, hogy bizonyos hiperparamétereket továbbra sem lehet gépi
tanulással megválasztani: az aktivációs függvényt, a rétegek számát és méretét,
a tanulási rátát, a jellemzőket és a regularizálást. A mély hálók sokszor nem
hatékonyak, mert egyes idegsejtek csak bizonyos tartományokban aktívak (Minsky
és Papert 1969).

A szóvektorokról szóló részben, jól nevelt öreg profhoz méltó módon elmondta a
2013-as cikkeket: múlt időben mondta, hogy _they were quite popular_. Az
skip-gram hasonló eredményt ér el, mint a folytonos szózsák (cbow), és
gyorsabban teszi. Az pontos implementáció körüli kontraverzióra, sőt egyáltalán
a két architektúra közötti különbségre hasonlóan tekint, mint a korábban
említett hiperparaméterekre: ki kell próbálni a különféle lehetőségeket.  Ezzel
szemben a korpusz megválasztására hívta fel a figyelmet.

A Mester külsőre is nagyon más volt, mint amire számítottam: kis termetű,
kicsit lányos. Egy régi kollégának az volt róla a tapasztalata, hogy elég
autistán ad elő: a laptopjába motyog. Nekem nem volt bajom az előadói
stílusával. Bár amikor a szünetben is odamentek hozzá kérdezni, akkor látszott,
hogy nagyon nehéz neki ez a helyzet, a padlót nézve hallgatta a kérdezőket,
mint aki gyóntat. Mindenesetre izgalommal várom a folytatást.

A második idősávjukban Piotr Bojanowski beszélt a fastText szó alatti
modelljeiről. Ha jól értem, a lengyel példa egy 'leananászoz' (mármint
ananásznak nevez) jelentésű lengyel szó volt. A fastText ismertetését bőszen
jegyzeteltem, például nem tudtam, hogy a karakter-ngramok hashelve vannak a
nagy embeddingmátrix elkerülésére. Újabb fejleményeket is elmondott (Mikolov+
LREC 18, Grave+ 18, Joulin+ fastText.zip 16). Folyt köv.

A második hármasból Bertrand Thirion (INRIA) kurzusát választottam: az agy
megértése gépi tanulással. A kognitív idegtudományi előadás olyan kutatásokról
szólt, ahol mesterséges ideghálókkal modellezik az agyi látást. Az mesterséges
háló aktivációiból jósolják az fMRI-képet. Ahol a jóslás pontos, ott
feltételezik, hogy az adott agyi terület hasonló számítást végez, mint a
mesterséges háló megfelelő része. Így azt is megállapítják, hogy a különböző
területek a feldolgozás milyen szintjéért felelnek . Validálták is a
kísérleteiket korábbi módszerekkel, amikből tudni lehet, hogy a látókéreg mely
területei felelnek a látvány közepéért és pereméért. A mesterséges hálónak a
színeket illetve a mozgást feldolgozó részei feletek meg ezeknek a
területeknek. Ez értelmesnek tűnik, hiszen az éles látásban inkább látjuk a
színeket, a szélén pedig mozgásokra figyelünk fel. http://nilearn.github.io/
Elképzelt képet is próbáltak kiolvas, aminek a mentális reprezentációra
nézve is tanulsága van: az öt körből álló stimulust, amik a dobókockán szokásos
módon voltak elhelyezve, a kísérleti személy X-alakú dologként írta le, és a
képen látszott is, hogy X alakban össze vannak kötve a pöttyök. Ennek az
előadásnak a második régészetről eljöttem, mert nagyon fáradt voltam, kicsit
rosszul is voltam, és a szállásra kellett jönnöm intézni valamit, de sajnálom,
mert olyasmiket ígért, hogy a dekódolás GATE-val való javítása, a hallás
dekódolása, és nyelvi terület.

A kínálat harmadik előadása generatív modellekről szól (Aaron Courville).
Rögtön a motiváló diát nem értettem, viszont megismertem számomra új
modelleket: az autoregresszív CNN-t (Van der Oord 16), és megértettem az
összefüggést a hanghullám hierarchikus szerkezete és a WaveNet tágított
(dilated) konvolúciója között. Az eloszlások közelítésén alapuló módszereket is
érteni véltem valamilyen szinten, büszke voltam magamra.
