// Run in the browser console (or via the browser-automation tool) with the page served from the repository root, e.g.
//   python -m http.server 8765   then open   http://127.0.0.1:8765/verify_in_browser/index.html
// Compares the BigInt core (window.PackVerify) with the Python verifier's exact outputs stored in certificates/*.verify_fraction.json
// and checks that tampered copies and Packomania's own (12-decimal) listing are rejected at zero tolerance.
(async () => {
  const get = async (u) => await (await fetch(u)).text();
  const out = {};
  for (const [n, url] of [[26, '/certificates/sum_radii_n26.json'], [120, '/certificates/A_n120.json'], [250, '/certificates/A_n250.json']]) {
    const r = PackVerify.verifyCertificateText(await get(url), { shrink: '1e-12' });
    const py = JSON.parse(await get(url.replace('.json', '.verify_fraction.json'))).reports[0];
    out['cert_' + n] = { valid: r.valid, browser_certified: r.certified, python_certified: py.certified_score.slice(0, 24),
                         agree_24_digits: r.certified.slice(0, 24) === py.certified_score.slice(0, 24) };  // python rounds, browser truncates: compare 24 chars
  }
  let obj = JSON.parse(await get('/certificates/A_n120.json'));
  obj.circles[7][2] = (parseFloat(obj.circles[7][2]) + 1e-9).toFixed(20);
  out.tampered_radius_valid = PackVerify.verifyCertificateText(JSON.stringify(obj), { shrink: '1e-12' }).valid;      // expected false
  obj = JSON.parse(await get('/certificates/A_n120.json')); obj.circles[117] = obj.circles[3].slice();
  out.tampered_duplicate_valid = PackVerify.verifyCertificateText(JSON.stringify(obj), { shrink: '1e-12' }).valid;   // expected false
  for (const n of [120, 250]) {
    const r = PackVerify.verifyCertificateText(await get(`/submission/csqv${n}.pck`), {});
    out['pck_' + n] = { valid: r.valid, sum: r.certified, contacts: r.contactsWithinTol, threeN: r.threeN };
  }
  const listed = PackVerify.verifyCertificateText(await get('/data/packomania_files/csqv120_as.pck'), {});
  out.packomania_listed_120_zero_tolerance = { valid: listed.valid, violations: listed.violations };                    // expected false
  return JSON.stringify(out, null, 1);
})();
