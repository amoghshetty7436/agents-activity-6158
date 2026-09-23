use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub prerelease: Option<String>,
    pub build: Option<String>,
}

pub fn parse(s: &str) -> Result<Version, String> {
    if s.is_empty() {
        return Err(format!("{} is not valid SemVer string", s));
    }

    let mut rest = s;

    // Build metadata (starts with +)
    let build = if let Some(pos) = rest.rfind('+') {
        let b = &rest[pos + 1..];
        if b.is_empty() {
            return Err(format!("{} is not valid SemVer string", s));
        }
        for id in b.split('.') {
            if id.is_empty() {
                return Err(format!("{} is not valid SemVer string", s));
            }
            for c in id.chars() {
                if !c.is_ascii_alphanumeric() && c != '-' {
                    return Err(format!("{} is not valid SemVer string", s));
                }
            }
        }
        rest = &rest[..pos];
        Some(b.to_string())
    } else {
        None
    };

    // Prerelease (starts with -)
    let prerelease = if let Some(pos) = rest.find('-') {
        let p = &rest[pos + 1..];
        if p.is_empty() {
            return Err(format!("{} is not valid SemVer string", s));
        }
        for id in p.split('.') {
            if id.is_empty() {
                return Err(format!("{} is not valid SemVer string", s));
            }
            // Check valid identifier
            // 0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*
            let mut is_numeric = true;
            let mut chars = id.chars();
            let first = match chars.next() {
                Some(c) => c,
                None => return Err(format!("{} is not valid SemVer string", s)),
            };
            if !first.is_ascii_alphanumeric() && first != '-' {
                return Err(format!("{} is not valid SemVer string", s));
            }
            if !first.is_ascii_digit() {
                is_numeric = false;
            } else if id.len() > 1 && first == '0' {
                // leading zero with letters/hyphen afterwards? Wait:
                // SemVer rule: numeric identifiers cannot have leading zeros ("01" invalid).
                // But non-numeric identifiers (containing letters/hyphens) CAN have leading zeros or arbitrary digits before letters/hyphens ("0A" ok).
                // Let's check Python regex: `0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*`
                // If it starts with a digit, is it purely numeric or does it contain a letter/hyphen?
            }

            let mut has_alpha_or_hyphen = false;
            for c in id.chars() {
                if c.is_ascii_alphabetic() || c == '-' {
                    has_alpha_or_hyphen = true;
                } else if !c.is_ascii_digit() {
                    return Err(format!("{} is not valid SemVer string", s));
                }
            }

            if !has_alpha_or_hyphen {
                // purely numeric identifier
                if id.len() > 1 && id.starts_with('0') {
                    return Err(format!("{} is not valid SemVer string", s));
                }
            } else {
                // must match `0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*`
                // Let's check regex validity via simple helper or logic:
                // `0` or `[1-9]\d*` or `\d*[a-zA-Z-][0-9a-zA-Z-]*`
                if !is_valid_prerelease_id(id) {
                    return Err(format!("{} is not valid SemVer string", s));
                }
            }
        }
        rest = &rest[..pos];
        Some(p.to_string())
    } else {
        None
    };

    // Core version: major.minor.patch
    let parts: Vec<&str> = rest.split('.').collect();
    if parts.len() != 3 {
        return Err(format!("{} is not valid SemVer string", s));
    }

    let major = parse_numeric_part(parts[0], s)?;
    let minor = parse_numeric_part(parts[1], s)?;
    let patch = parse_numeric_part(parts[2], s)?;

    Ok(Version {
        major,
        minor,
        patch,
        prerelease,
        build,
    })
}

fn is_valid_prerelease_id(id: &str) -> bool {
    if id == "0" {
        return true;
    }
    // Check if it's [1-9]\d*
    if id.chars().all(|c| c.is_ascii_digit()) {
        return !id.starts_with('0') && !id.is_empty();
    }
    // Otherwise must be \d*[a-zA-Z-][0-9a-zA-Z-]*
    let mut chars = id.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            chars.next();
        } else {
            break;
        }
    }
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '-' => {
            for c in chars {
                if !c.is_ascii_alphanumeric() && c != '-' {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}

fn parse_numeric_part(part: &str, full: &str) -> Result<u64, String> {
    if part.is_empty() {
        return Err(format!("{} is not valid SemVer string", full));
    }
    if part.len() > 1 && part.starts_with('0') {
        return Err(format!("{} is not valid SemVer string", full));
    }
    for c in part.chars() {
        if !c.is_ascii_digit() {
            return Err(format!("{} is not valid SemVer string", full));
        }
    }
    part.parse::<u64>().map_err(|_| format!("{} is not valid SemVer string", full))
}

pub fn to_string(v: &Version) -> String {
    let mut s = format!("{}.{}.{}", v.major, v.minor, v.patch);
    if let Some(ref pre) = v.prerelease {
        s.push('-');
        s.push_str(pre);
    }
    if let Some(ref bld) = v.build {
        s.push('+');
        s.push_str(bld);
    }
    s
}

pub fn compare(a: &Version, b: &Version) -> Ordering {
    let ord = a.major.cmp(&b.major);
    if ord != Ordering::Equal {
        return ord;
    }
    let ord = a.minor.cmp(&b.minor);
    if ord != Ordering::Equal {
        return ord;
    }
    let ord = a.patch.cmp(&b.patch);
    if ord != Ordering::Equal {
        return ord;
    }

    match (&a.prerelease, &b.prerelease) {
        (None, None) => Ordering::Equal,
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (Some(rc_a), Some(rc_b)) => nat_cmp(rc_a, rc_b),
    }
}

fn nat_cmp(a: &str, b: &str) -> Ordering {
    let a_parts: Vec<&str> = a.split('.').collect();
    let b_parts: Vec<&str> = b.split('.').collect();

    for (sub_a, sub_b) in a_parts.iter().zip(b_parts.iter()) {
        let res = cmp_prerelease_tag(sub_a, sub_b);
        if res != Ordering::Equal {
            return res;
        }
    }

    a_parts.len().cmp(&b_parts.len())
}

fn cmp_prerelease_tag(a: &str, b: &str) -> Ordering {
    let a_num = a.chars().all(|c| c.is_ascii_digit()) && !a.is_empty();
    let b_num = b.chars().all(|c| c.is_ascii_digit()) && !b.is_empty();

    if a_num && b_num {
        let na = a.parse::<u64>().unwrap_or(0);
        let nb = b.parse::<u64>().unwrap_or(0);
        na.cmp(&nb)
    } else if a_num {
        Ordering::Less
    } else if b_num {
        Ordering::Greater
    } else {
        a.cmp(b)
    }
}

pub fn bump_major(v: &Version) -> Version {
    Version {
        major: v.major + 1,
        minor: 0,
        patch: 0,
        prerelease: None,
        build: None,
    }
}

pub fn bump_minor(v: &Version) -> Version {
    Version {
        major: v.major,
        minor: v.minor + 1,
        patch: 0,
        prerelease: None,
        build: None,
    }
}

pub fn bump_patch(v: &Version) -> Version {
    Version {
        major: v.major,
        minor: v.minor,
        patch: v.patch + 1,
        prerelease: None,
        build: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_and_to_string() {
        let v = parse("1.2.3-alpha.1+build.123").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
        assert_eq!(v.prerelease.as_deref(), Some("alpha.1"));
        assert_eq!(v.build.as_deref(), Some("build.123"));
        assert_eq!(to_string(&v), "1.2.3-alpha.1+build.123");
    }

    #[test]
    fn test_invalid_parse() {
        assert!(parse("01.2.3").is_err());
        assert!(parse("1.0").is_err());
        assert!(parse("1.2.3-").is_err());
        assert!(parse("1.2.3-01").is_err());
    }

    #[test]
    fn test_compare() {
        let v1 = parse("1.0.0-alpha").unwrap();
        let v2 = parse("1.0.0-alpha.1").unwrap();
        let v3 = parse("1.0.0-alpha.beta").unwrap();
        let v4 = parse("1.0.0-beta").unwrap();
        let v5 = parse("1.0.0-beta.2").unwrap();
        let v6 = parse("1.0.0-beta.11").unwrap();
        let v7 = parse("1.0.0-rc.1").unwrap();
        let v8 = parse("1.0.0").unwrap();

        assert_eq!(compare(&v1, &v2), Ordering::Less);
        assert_eq!(compare(&v2, &v3), Ordering::Less);
        assert_eq!(compare(&v3, &v4), Ordering::Less);
        assert_eq!(compare(&v4, &v5), Ordering::Less);
        assert_eq!(compare(&v5, &v6), Ordering::Less);
        assert_eq!(compare(&v6, &v7), Ordering::Less);
        assert_eq!(compare(&v7, &v8), Ordering::Less);
        assert_eq!(compare(&v8, &v7), Ordering::Greater);
    }

    #[test]
    fn test_bumps() {
        let v = parse("1.2.3-rc.1+build").unwrap();
        assert_eq!(to_string(&bump_major(&v)), "2.0.0");
        assert_eq!(to_string(&bump_minor(&v)), "1.3.0");
        assert_eq!(to_string(&bump_patch(&v)), "1.2.4");
    }
}
